"""Load models and datasets from the HuggingFace hub."""

import logging
import time

import datasets
import giskard as gsk
import huggingface_hub
import torch
from giskard import Dataset
from giskard.models.base import BaseModel
from giskard.models.huggingface import HuggingFaceModel
from transformers.pipelines import TextClassificationPipeline

from .base_loader import BaseLoader, DatasetError

logger = logging.getLogger(__name__)


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

    def load_giskard_model_dataset(self, model_id, dataset=None, dataset_config=None, dataset_split=None):
        # If no dataset was provided, we try to get it from the model metadata.
        if dataset is None:
            logger.debug("No dataset provided. Trying to get it from the model metadata.")
            dataset = self._find_dataset_id_from_model(model_id)
            logger.debug(f"Found dataset `{dataset}`.")

        # Loading the model is easy. What is complicated is to get the dataset.
        # So we start by trying to get the dataset, because if we fail, we don't
        # want to waste time downloading the model.
        hf_dataset = self.load_dataset(dataset, dataset_config, dataset_split, model_id)

        # Load the model.
        hf_model = self.load_model(model_id)

        # Check that the dataset has the good feature names for the task.
        feature_mapping = self._get_feature_mapping(hf_model, hf_dataset)

        df = hf_dataset.to_pandas().rename(columns={v: k for k, v in feature_mapping.items()})

        # @TODO: currently for classification models only.
        id2label = hf_model.model.config.id2label
        df["label"] = df.label.apply(lambda x: id2label[x])

        gsk_dataset = gsk.Dataset(df, target="label", column_types={"text": "text"})

        gsk_model = HuggingFaceModel(
            hf_model,
            model_type="classification",
            data_preprocessing_function=lambda df: df.text.tolist(),
            classification_labels=[id2label[i] for i in range(len(id2label))],
            batch_size=None,
            device=self.device,
        )

        # Optimize batch size
        if self.device.startswith("cuda"):
            gsk_model.batch_size = self._find_optimal_batch_size(gsk_model, gsk_dataset)

        return gsk_model, gsk_dataset

    def load_dataset(self, dataset_id, dataset_config=None, dataset_split=None, model_id=None):
        """Load a dataset from the HuggingFace Hub."""
        logger.debug(f"Trying to load dataset `{dataset_id}` (config = `{dataset_config}`, split = `{dataset_split}`).")
        try:
            hf_dataset = datasets.load_dataset(dataset_id, name=dataset_config, split=dataset_split)

            if dataset_split is None:
                dataset_split = self._select_best_dataset_split(list(hf_dataset.keys()))
                logger.debug(f"No split provided, automatically selected split = `{dataset_split}`).")
                hf_dataset = hf_dataset[dataset_split]

            return hf_dataset
        except ValueError as err:
            msg = f"Could not load dataset `{dataset_id}` with config `{dataset_config}`."
            raise DatasetError(msg) from err

    def load_model(self, model_id):
        from transformers import pipeline

        task = huggingface_hub.model_info(model_id).pipeline_tag

        return pipeline(task=task, model=model_id, device=self.device)

    def _get_feature_mapping(self, hf_model, hf_dataset):
        if isinstance(hf_model, TextClassificationPipeline):
            task_features = {"text": "string", "label": "class_label"}
        else:
            msg = "Unsupported model type."
            raise NotImplementedError(msg)

        dataset_features = hf_dataset.features
        feature_mapping = {f: f for f in set(task_features) & set(dataset_features)}

        missing_features = set(task_features) - set(feature_mapping)

        if not missing_features:
            return feature_mapping

        # If not, we try to find a suitable mapping by matching types.
        available_features = set(dataset_features) - set(feature_mapping)
        for feature in missing_features:
            expected_type = task_features[feature]
            if expected_type == "class_label":
                candidates = [f for f in available_features if isinstance(dataset_features[f], datasets.ClassLabel)]
            else:
                candidates = [f for f in available_features if dataset_features[f].dtype == expected_type]

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

                ds_slice = dataset.slice(lambda df: df.sample(num_samples), row_level=False)

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
