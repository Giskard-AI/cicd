"""Load models and datasets from Github."""

import logging
from abc import ABC, abstractmethod

from giskard.models.base import BaseModel
from giskard.core.model_validation import validate_model
from giskard import Dataset

logger = logging.getLogger(__name__)


class LoaderError(RuntimeError):
    """Could not load the model and/or dataset."""


class DatasetError(LoaderError):
    """Problems related to the dataset."""


class ModelError(LoaderError):
    """Problems related to the model."""


class BaseLoader(ABC):

    @abstractmethod
    def load_giskard_model_dataset(self) -> (BaseModel, Dataset):
        ...

    def validate(self):
        gsk_model, gsk_dataset = self.load_giskard_model_dataset()
        validate_model(gsk_model, gsk_dataset)
