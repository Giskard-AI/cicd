import giskard
import yaml

path_to_config = __file__.split("train.py")[0]+"cicd_config.yaml"
with open(path_to_config) as yaml_f:
    cicd_config = yaml.load(yaml_f, Loader=yaml.Loader)

# Replace this with your own data & model creation.
df = giskard.demo.titanic_df()
data_preprocessor, clf = giskard.demo.titanic_pipeline()

# Wrap your Pandas DataFrame with Giskard.Dataset (test set, a golden dataset, etc.). Check the dedicated doc page: https://docs.giskard.ai/en/latest/guides/wrap_dataset/index.html
giskard_dataset = giskard.Dataset(
    df=df,  # A pandas.DataFrame that contains the raw data (before all the pre-processing steps) and the actual ground truth variable (target).
    target="Survived",  # Ground truth variable
    name="Titanic dataset", # Optional
    cat_columns=['Pclass', 'Sex', "SibSp", "Parch", "Embarked"]  # Optional, but is a MUST if available. Inferred automatically if not.
)

# Wrap your model with Giskard.Model. Check the dedicated doc page: https://docs.giskard.ai/en/latest/guides/wrap_model/index.html
# you can use any tabular, text or LLM models (PyTorch, HuggingFace, LangChain, etc.),
# for classification, regression & text generation.
def prediction_function(df):
    # The pre-processor can be a pipeline of one-hot encoding, imputer, scaler, etc.
    preprocessed_df = data_preprocessor(df)
    return clf.predict_proba(preprocessed_df)

giskard_model = giskard.Model(
    model=prediction_function,  # A prediction function that encapsulates all the data pre-processing steps and that could be executed with the dataset used by the scan.
    model_type="classification",  # Either regression, classification or text_generation.
    name="Titanic model",  # Optional
    classification_labels=clf.classes_,  # Their order MUST be identical to the prediction_function's output order
    feature_names=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'],  # Default: all columns of your dataset
    # classification_threshold=0.5,  # Default: 0.5
)

from giskard_cicd.utils import dump_model_and_dataset_for_cicd
dump_model_and_dataset_for_cicd(cicd_config["artifact_path"], giskard_model, giskard_dataset)
