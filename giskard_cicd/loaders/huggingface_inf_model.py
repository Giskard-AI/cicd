import logging
import os

import numpy as np
import pandas as pd
from ..automation.hf_api import request_inf_api
from giskard import Model

logger = logging.getLogger(__file__)



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
    
    # Utitlity to extract scores
    def extract_scores(data):
        if isinstance(data, dict):
            return [data.get('score')] + sum((extract_scores(v) for v in data.values()), [])
        elif isinstance(data, list):
            return sum((extract_scores(v) for v in data), [])
        else:
            return []
    
    # Utility to query HF inference API
    def query(payload):
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError(
                "Missing Hugging Face access token. Please provide it in `HF_TOKEN` environment variable"
            )

        hf_inference_api_endpoint = os.environ.get(
            "HF_INFERENCE_ENDPOINT", default="https://api-inference.huggingface.co"
        )
        headers = {"Authorization": f"Bearer {hf_token}"}
        return request_inf_api(hf_inference_api_endpoint, model_name, headers, payload)

    # Text classification: limit the scope so that the model does not import giskard_cicd
    def predict_from_text_classification_inference(df: pd.DataFrame) -> np.ndarray:
        results = []
        # get all text from the dataframe
        raw_inputs = df["text"].tolist()

        for i in range(0, len(raw_inputs), inference_api_batch_size):
            inputs = raw_inputs[i : min(i + inference_api_batch_size, len(raw_inputs))]
            payload = {"inputs": inputs, "options": {"use_cache": True}}

            logger.debug(
                f"Requesting {len(inputs)} rows of data: ({i}/{len(raw_inputs)})"
            )

            outputs = query(payload)

            for output in outputs:
                results.append(extract_scores(output))

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
