import logging
import os
import re
from time import sleep

import numpy as np
import pandas as pd
from requests_toolbelt import sessions
from giskard import Model

logger = logging.getLogger(__name__)


def classification_model_from_inference_api(
    model_name,
    labels,
    features,
    model_type="text_classification",
    inference_api_batch_size=200,
):
    """
    Get a Giskard model from the HuggingFace inference API.
    """
    if model_type != "text_classification":
        raise NotImplementedError(
            f"Not supported model type: {model_type}. Only text_classification models are supported for now."
        )

    # Allow to customize the HF API endpoint and reuse a session
    # See https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfinferenceendpoint
    session = sessions.BaseUrlSession(
        os.environ.get(
            "HF_INFERENCE_ENDPOINT", default="https://api-inference.huggingface.co"
        )
    )
    url = f"models/{model_name}"
    # Create a basic template for request payload
    request_payload = {"options": {"use_cache": True}}

    def extract_inference_api_max_length(error_info):
        max_length = 512  # Set a default value that most model uses
        matched = re.search("must match the size of tensor b \((\d+)\)", error_info)
        if matched:
            try:
                max_length = int(matched.group(1))
                return max_length
            except Exception as e:
                logger.debug(e)

        logger.warning(
            f"Failed to extract max input length, use {max_length} by default"
        )
        return max_length

    # Utility to query HF inference API
    def query(payload):
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "Missing Hugging Face access token. Please provide it in `HF_TOKEN` environment variable"
            )

        if not session.base_url:
            # GSK-3373: after unpickle, `base_url` is lost in the session
            # See https://github.com/requests/toolbelt/issues/375
            # We reset it for flexibility
            session.base_url = os.environ.get(
                "HF_INFERENCE_ENDPOINT", default="https://api-inference.huggingface.co"
            )

        headers = {"Authorization": f"Bearer {hf_token}"}
        response = session.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            logger.debug(f"Request to inference API returns {response.status_code}")
        try:
            return response.status_code, response.json()
        except Exception:
            return response.status_code, {"error": response.content}

    def handle_hf_inference_api_error(status_code, output):
        if status_code == 200:
            return

        logger.debug(
            f'Handling {output["error"] if "error" in output else "unknown"} error with {status_code}'
        )

        if "warnings" in output and "Input is too long" in output["error"]:
            """
            When input is too long:
            {
                'error': 'Input is too long, try to truncate or use a paramater'
                    + 'to handle this: The size of tensor a (702) must match the'
                    + 'size of tensor b (512) at non-singleton dimension 1',
                'warnings': [...],
            }
            """
            if (
                "parameters" in request_payload
                and "truncation" in request_payload["parameters"]
                and request_payload["parameters"]
            ):
                if "max_length" in request_payload["parameters"]:
                    # The model still cannot handle the input
                    raise ValueError(
                        f"HF inference API cannot handle too long input: {output['error']}"
                    )
                # Patch max length
                request_payload["parameters"].update(
                    {"max_length": extract_inference_api_max_length(output["error"])}
                )
                logger.warning(
                    "Your model is missing a max length in tokenizer config. "
                    f"We are using {request_payload['parameters']['max_length']} as an alternative. "
                    'Please add `"model_max_length": 512` to your `tokenizer_config.json` file.'
                )
            else:
                # Enable truncation for the request
                request_payload.update({"parameters": {"truncation": True}})
        elif status_code == 404:
            raise RuntimeError("Model not found")

    # Text classification: limit the scope so that the model does not import giskard_cicd
    def predict_from_text_classification_inference(df: pd.DataFrame) -> np.ndarray:
        results = []
        # get all text from the dataframe
        raw_inputs = df["text"].tolist()

        for i in range(0, len(raw_inputs), inference_api_batch_size):
            inputs = raw_inputs[i : min(i + inference_api_batch_size, len(raw_inputs))]
            request_payload.update(
                {"inputs": inputs}
            )  # Reuse the template to avoid adding parameters several times

            logger.debug(
                f"Requesting {len(inputs)} rows of data: ({i}/{len(raw_inputs)})"
            )
            # TODO(Inoki): Make it configurable or choose a proper value
            max_retries = 100
            retry = max_retries
            output = {"error": "First attemp"}
            while retry > 0 and "error" in output:
                # Retry after 1 second
                sleep(1)
                status_code, output = query(request_payload)

                if "error" in output:
                    logger.debug(f"Request returns an error with {status_code}")
                    handle_hf_inference_api_error(status_code, output)
                else:
                    logger.debug(f"Request returns {status_code}")

                retry -= 1
                if retry <= 0:
                    raise RuntimeError(
                        f"Failed to request {url} after {max_retries} retries"
                    )

            for single_output in output:
                try:
                    sorted_output = sorted(
                        single_output, key=lambda x: labels.index(x["label"])
                    )
                    results.append([x["score"] for x in sorted_output])
                except Exception as e:
                    logger.error(
                        f"Unexpected format of output: {single_output}, {e}, {labels}"
                    )
                    results.append([-0.1] * len(labels))

        logger.debug(f"Finished, got {len(results)} results")

        return np.array(results)

    return Model(
        model=lambda df: predict_from_text_classification_inference(
            df
        ),  # A prediction function that encapsulates all the data pre-processing steps and that
        model_type="classification",  # Either regression, classification or text_generation.
        name=f"{model_name} HF inference API",  # Optional
        classification_labels=labels,  # Their order MUST be identical to the prediction_function's
        feature_names=features,  # Default: all columns of your dataset
    )
