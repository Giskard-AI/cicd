from giskard import Model
import pandas as pd
import numpy as np
from time import sleep

def extract_scores(data):
    scores = []
    
    if isinstance(data, dict):
        # extract 'score' value if it exists
        score = data.get('score')
        if score is not None:
            scores.append(score)
        
        # Recursively process dictionary values
        for value in data.values():
            scores.extend(extract_scores(value))
    
    elif isinstance(data, list):
        # recursively process list elements
        for element in data:
            scores.extend(extract_scores(element))
    
    return scores

def predict_from_inference(df: pd.DataFrame, query) -> np.ndarray:
    results = []
    # get all text from the dataframe
    inputs = df["text"].tolist()

    payload = {"inputs": inputs, "options": {"use_cache": True, "wait_for_model": True}}
    output = query(payload)
    sleep(0.5)

    for i in output:
        results.append(extract_scores(i))

    return np.array(results)

def classification_model_from_inference_api(model_name, labels, features, query):
    """Get a Giskard model from the HuggingFace inference API."""
    return Model(
        model = lambda df: predict_from_inference(df, query),  # A prediction function that encapsulates all the data pre-processing steps and that
        model_type="classification",  # Either regression, classification or text_generation.
        name=model_name,  # Optional
        classification_labels=labels,  # Their order MUST be identical to the prediction_function's
        feature_names=features  # Default: all columns of your dataset
    )
