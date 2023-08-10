import pandas as pd
import cloudpickle
from pathlib import Path

import giskard
from giskard import Dataset
from giskard.models.base import BaseModel

from giskard_cicd.loaders.base_loader import BaseLoader


class MyGithubLoader(BaseLoader):
    def load_giskard_model_dataset(self, model, dataset) -> (BaseModel, Dataset):

        # Loading the model you want to include in CICD
        model_path = Path(model)
        if model_path.exists():
            with open(model_path, "rb") as f:
                my_model = cloudpickle.load(f)
        else:
            raise ValueError(
                "We couldn't load your model with cloudpickle. Please provide us with your own "
                "serialisation method by overriding the save_model() and load_model() methods."
            )

        # Loading the dataset you want to use to scan your model during the CICD pipeline
        my_dataset = pd.read_csv(
            dataset,
            keep_default_na=False,
            na_values=["_GSK_NA_"],
        )

        # Wrap your Pandas DataFrame with Giskard.Dataset (test set, a golden dataset, etc.).
        # Check the dedicated doc page: https://docs.giskard.ai/en/latest/guides/wrap_dataset/index.html
        giskard_dataset = giskard.Dataset(
            df=my_dataset,
            # A pandas.DataFrame that contains the raw data (before all the pre-processing steps)
            # and the actual ground truth variable (target).
            target="Survived",  # Ground truth variable
            name="Titanic dataset",  # Optional
            cat_columns=['Pclass', 'Sex', "SibSp", "Parch", "Embarked"]
            # Optional, but is a MUST if available. Inferred automatically if not.
        )

        # Wrap your model with Giskard.Model.
        # Check the dedicated doc page: https://docs.giskard.ai/en/latest/guides/wrap_model/index.html
        # you can use any tabular, text or LLM models (PyTorch, HuggingFace, LangChain, etc.),
        # for classification, regression & text generation.
        def prediction_function(df):
            return my_model.predict_proba(df)

        giskard_model = giskard.Model(
            model=prediction_function,
            # A prediction function that encapsulates all the data pre-processing steps
            # and that could be executed with the dataset used by the scan.
            model_type="classification",  # Either regression, classification or text_generation.
            name="Titanic model",  # Optional
            classification_labels=my_model.classes_,
            # Their order MUST be identical to the prediction_function's output order
            feature_names=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],
            # Default: all columns of your dataset
            # classification_threshold=0.5,  # Default: 0.5
        )

        return giskard_model, giskard_dataset
