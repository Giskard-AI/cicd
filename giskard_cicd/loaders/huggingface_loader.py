"""Load models and datasets from the HuggingFace hub."""

import logging
import time
from typing import Dict

import datasets
import giskard as gsk
import huggingface_hub
import pandas as pd
import os
import torch
from giskard import Dataset
from giskard.models.base import BaseModel
from giskard.models.huggingface import HuggingFaceModel
from transformers.pipelines import TextClassificationPipeline
import numpy as np

from .base_loader import BaseLoader, DatasetError
from .huggingface_inf_model import classification_model_from_inference_api

logger = logging.getLogger(__file__)


class HuggingFacePipelineModel(HuggingFaceModel):
    def _get_predictions(self, data):
        # Override _get_predictions method, which is called by HuggingFaceModel
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")
        _predictions = [
            {p["label"]: p["score"] for p in pl}
            for pl in self.model(data, top_k=None, truncation=True)
        ]
        return [
            [p[label] for label in self.classification_labels] for p in _predictions
        ]


class HuggingFaceLoader(BaseLoader):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _find_dataset_id_from_model(self, model_id):
        """Find the dataset ID from the model metadata."""
        model_card = huggingface_hub.model_info(model_id).cardData

        if "datasets" not in model_card:
            msg = f"Could not find dataset for model `{model_id}`."
            raise DatasetError(msg)

        # Take the first one
        dataset_id = model_card["datasets"][0]
        return dataset_id

    def load_giskard_model_dataset(
        self,
        model=None,
        dataset=None,
        dataset_config=None,
        dataset_split=None,
        manual_feature_mapping: Dict[str, str] = None,
        classification_label_mapping: Dict[int, str] = None,
        hf_token=None,
        inference_type="hf_pipeline",
        inference_api_token=None,
        inference_api_batch_size=200,
    ):
        model_id = model
        # If no dataset was provided, we try to get it from the model metadata.
        if dataset is None:
            logger.debug(
                "No dataset provided. Trying to get it from the model metadata."
            )
            dataset = self._find_dataset_id_from_model(model_id)
            logger.debug(f"Found dataset `{dataset}`.")

        # Loading the model is easy. What is complicated is to get the dataset.
        # So we start by trying to get the dataset, because if we fail, we don't
        # want to waste time downloading the model.
        hf_dataset = self.load_dataset(dataset, dataset_config, dataset_split, model_id)
        # Flatten dataset to avoid `datasets.DatasetDict`
        hf_dataset = self._flatten_hf_dataset(hf_dataset, dataset_split)

        if isinstance(hf_dataset, datasets.Dataset):
            logger.debug(f"Loaded dataset with {hf_dataset.size_in_bytes} bytes")
        else:
            logger.warning("Loaded dataset is not a Dataset object, the scan may fail.")

        hf_model = None

        # Check that the dataset has the good feature names for the task.
        logger.debug("Retrieving feature mapping")
        if manual_feature_mapping is None:
            hf_model = self.load_model(model_id)
            feature_mapping = self._get_feature_mapping(hf_model, hf_dataset)
            logger.warn(
                f'Feature mapping is not provided, using extracted "{feature_mapping}"'
            )
        else:
            feature_mapping = manual_feature_mapping

        df = hf_dataset.to_pandas().rename(
            columns={v: k for k, v in feature_mapping.items()}
        )

        # remove the rows have multiple labels
        # this is a hacky way to do it
        # we do not support multi-label classification for now
        if "label" in df and isinstance(df.label[0], list):
            df = df[df.apply(lambda row: len(row["label"]) == 1, axis=1)]

        logger.debug(f"Overview of dataset: `{dataset}`.")

        # @TODO: currently for classification models only.
        logger.debug("Retrieving classification label mapping")
        if classification_label_mapping is None:
            if hf_model is None:
                hf_model = self.load_model(model_id)
            id2label = hf_model.model.config.id2label
            logger.warn(f'Label mapping is not provided, using "{id2label}" from model')
        else:
            id2label = classification_label_mapping

        label_keys = [k for k in df.keys() if k.startswith("label")]
        label_key = label_keys[0]

        if (
            label_key
            and isinstance(df[label_key][0], list)
            or isinstance(df[label_key][0], np.ndarray)
        ):
            # need to include all labels
            # rewrite this lambda function to include all labels
            df[label_key] = df[label_key].apply(lambda x: id2label[x[0]])
        else:
            # @TODO: when the label for test is not provided, what do we do?
            df[label_key] = df[label_key].apply(
                lambda x: id2label[x] if x >= 0 else "-1"
            )
        # map the list of label ids to the list of labels
        # df["label"] = df.label.apply(lambda x: [id2label[i] for i in x])
        logger.debug("Wrapping dataset")
        gsk_dataset = gsk.Dataset(
            df[:100],
            name=f"HF {dataset}[{dataset_config}]({dataset_split}) for {model_id} model",
            target="label",
            column_types={"text": "text"},
            validation=False,
        )

        logger.debug("Wrapping model")

        gsk_model = self._get_gsk_model(
            model_id,
            hf_model,
            [id2label[i] for i in range(len(id2label))],
            features=feature_mapping,
            inference_type=inference_type,
            device=self.device,
            hf_token=inference_api_token,
            inference_api_batch_size=inference_api_batch_size,
        )

        # Optimize batch size
        if self.device.startswith("cuda"):
            gsk_model.batch_size = self._find_optimal_batch_size(gsk_model, gsk_dataset)

        return gsk_model, gsk_dataset

    def load_dataset(
        self, dataset_id, dataset_config=None, dataset_split=None, model_id=None
    ):
        """Load a dataset from the HuggingFace Hub."""
        logger.debug(
            f"Trying to load dataset `{dataset_id}` (config = `{dataset_config}`, split = `{dataset_split}`)."
        )
        try:
            # we do not set the split here
            # because we want to be able to select the best split later with preprocessing
            hf_dataset = datasets.load_dataset(dataset_id, name=dataset_config)

            if isinstance(hf_dataset, datasets.Dataset):
                logger.debug(f"Loaded dataset with {hf_dataset.size_in_bytes} bytes")
            else:
                logger.debug("Loaded dataset is a DatasetDict")

            if dataset_split is None:
                dataset_split = self._select_best_dataset_split(list(hf_dataset.keys()))
                logger.info(
                    f"No split provided, automatically selected split = `{dataset_split}`)."
                )
                hf_dataset = hf_dataset[dataset_split]

            return hf_dataset
        except ValueError as err:
            msg = (
                f"Could not load dataset `{dataset_id}` with config `{dataset_config}`."
            )
            raise DatasetError(msg) from err

    def load_model(self, model_id):
        from transformers import pipeline

        task = huggingface_hub.model_info(model_id).pipeline_tag

        return pipeline(task=task, model=model_id, device=self.device)

    def _get_gsk_model(
        self,
        model_id,
        hf_model,
        labels,
        features=None,
        inference_type="hf_inference_api",
        device=None,
        hf_token=None,
        inference_api_batch_size=200,
    ):
        if hf_model is None and inference_type == "hf_pipeline":
            hf_model = self.load_model(model_id)
        logger.info(f"Loading '{inference_type}' model from Hugging Face")
        if inference_type == "hf_pipeline":
            return HuggingFacePipelineModel(
                hf_model,
                model_type="classification",
                name=f"{model_id} HF pipeline",
                data_preprocessing_function=lambda df: df.text.tolist(),
                classification_labels=labels,
                batch_size=None,
                device=device,
                feature_names=["text"] if features is None else features,
            )
        elif inference_type == "hf_inference_api":
            if features is None:
                raise ValueError(
                    "features must be provided when using model_type='hf_inference_api'"
                )

            if hf_token is None:
                raise ValueError(
                    "hf_token must be provided when using model_type='hf_inference_api'"
                )
            # To be used later in inference API mmodel
            os.environ.update([("HF_TOKEN", hf_token)])

            return classification_model_from_inference_api(
                model_id,
                labels,
                features,
                inference_api_batch_size=inference_api_batch_size,
            )

    def _get_dataset_features(self, hf_dataset):
        """
        Recursively get the features of the dataset
        """
        dataset_features = {}
        try:
            dataset_features = hf_dataset.features
        except AttributeError:
            logger.warning("Features not found")
            if isinstance(hf_dataset, datasets.DatasetDict):
                keys = list(hf_dataset.keys())
                return self._get_dataset_features(hf_dataset[keys[0]])
        return dataset_features

    def _flatten_hf_dataset(self, hf_dataset, data_split=None):
        """
        Flatten the dataset to a pandas dataframe
        """
        flat_dataset = pd.DataFrame()
        if isinstance(hf_dataset, datasets.DatasetDict):
            keys = list(hf_dataset.keys())
            for k in keys:
                if data_split is not None and k == data_split:
                    # Match the data split
                    flat_dataset = hf_dataset[k]
                    break

                # Otherwise infer one data split
                if k.startswith("train"):
                    continue
                elif k.startswith(data_split):
                    # TODO: only support one split for now
                    # Maybe we can merge all the datasets into one
                    flat_dataset = hf_dataset[k]
                    break
                else:
                    flat_dataset = hf_dataset[k]

            # If there are only train datasets
            if isinstance(flat_dataset, pd.DataFrame) and flat_dataset.empty:
                flat_dataset = hf_dataset[keys[0]]

        return flat_dataset

    def _get_feature_mapping(self, hf_model, hf_dataset):
        if isinstance(hf_model, TextClassificationPipeline):
            task_features = {"text": "string", "label": "class_label"}
        else:
            msg = "Unsupported model type."
            raise NotImplementedError(msg)

        dataset_features = self._get_dataset_features(hf_dataset)
        # map features
        feature_mapping = {}
        for f in set(dataset_features):
            if f in task_features:
                feature_mapping[f] = f
            else:
                for t in task_features:
                    if f.startswith(t):
                        feature_mapping[t] = f

        if not set(task_features) - set(feature_mapping):
            return feature_mapping
        else:
            # If not, we try to find a suitable mapping by matching types.
            return self._amend_missing_features(
                task_features, dataset_features, feature_mapping
            )

    def _amend_missing_features(self, task_features, dataset_features, feature_mapping):
        """
        Question: what is this code doing?
        """
        available_features = set(dataset_features) - set(feature_mapping)
        missing_features = set(task_features) - set(feature_mapping)

        for feature in missing_features:
            expected_type = task_features[feature]
            if expected_type == "class_label":
                candidates = [
                    f
                    for f in available_features
                    if isinstance(dataset_features[f], datasets.ClassLabel)
                ]
            else:
                candidates = [
                    f
                    for f in available_features
                    if dataset_features[f].dtype == expected_type
                ]

            # If we have more than one match, it`s not possible to know which one is the good one.
            if len(candidates) != 1:
                msg = f"Could not find a suitable mapping for feature for `{feature}`."
                raise RuntimeError(msg)

            feature_mapping[feature] = candidates[0]
            available_features.remove(candidates[0])
        return feature_mapping

    def _select_best_dataset_split(self, split_names):
        """Get the best split for testing.

        Selects the split `test` if available, otherwise `validation`, and as a last resort `train`.
        If there is only one split, we return that split.
        """
        # If only one split is available, we just use that one.
        if len(split_names) == 1:
            return split_names[0]

        # Otherwise iterate based on the preferred prefixes.
        for prefix in ["test", "valid", "train"]:
            try:
                return next(x for x in split_names if x.startswith(prefix))
            except StopIteration:
                pass

        return None

    def _find_optimal_batch_size(self, model: BaseModel, dataset: Dataset):
        """Find the optimal batch size for the model and dataset."""
        initial_batch_size = model.batch_size
        try:
            model.batch_size = 1
            inference_time = float("inf")
            while True:
                num_runs = min(30, len(dataset) // model.batch_size)
                num_samples = num_runs * model.batch_size
                if num_runs == 0:
                    return model.batch_size // 2

                ds_slice = dataset.slice(
                    lambda df: df.sample(num_samples), row_level=False
                )

                t_start = time.perf_counter_ns()
                try:
                    with gsk.models.cache.no_cache():
                        model.predict(ds_slice)
                except RuntimeError:
                    return model.batch_size // 2
                elapsed = time.perf_counter_ns() - t_start

                time_per_sample = elapsed / (num_samples)
                if time_per_sample > inference_time:
                    return model.batch_size // 2
                inference_time = time_per_sample
                model.batch_size *= 2
        finally:
            model.batch_size = initial_batch_size
