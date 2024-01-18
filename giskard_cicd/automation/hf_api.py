
import requests
import logging
from time import sleep

logger = logging.getLogger(__file__)

def request_inf_api(hf_inference_api_endpoint, model_name, headers, payload):
    # Allow to customize the HF API endpoint
    # See https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfinferenceendpoint
    url = f"{hf_inference_api_endpoint}/models/{model_name}"
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        logger.debug(f"Request to inference API returns {response.status_code}")
        while response.status_code == 500:
            logger.debug(f"Retry request to {url}")
            sleep(0.5)
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                break
    try:
        return response.json()
    except Exception:
        return {"error": response.content}