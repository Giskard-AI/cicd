import yaml
from giskard import Dataset, Model
from giskard.models.base import BaseModel
from giskard.ml_worker.utils.file_utils import get_file_name

from .base_loader import BaseLoader
from pathlib import Path
from giskard.core.core import DatasetMeta


class GithubLoader(BaseLoader):
    #TODO: change the way dataset is loaded, factor out some of the logic contained in Dataset.download()
    def load_giskard_model_dataset(self, model, dataset) -> (BaseModel, Dataset):
        with open(Path(dataset) / "giskard-dataset-meta.yaml") as f:
            saved_meta = yaml.load(f, Loader=yaml.Loader)
            meta = DatasetMeta(
                name=saved_meta["name"],
                target=saved_meta["target"],
                column_types=saved_meta["column_types"],
                column_dtypes=saved_meta["column_dtypes"],
                number_of_rows=saved_meta["number_of_rows"],
                category_features=saved_meta["category_features"],
            )

        df = Dataset.load(Path(dataset) / get_file_name("data", "csv.zst", False))
        df = Dataset.cast_column_to_dtypes(df, meta.column_dtypes)

        return Model.load(model), Dataset(
            df=df,
            name=meta.name,
            target=meta.target,
            column_types=meta.column_types,
        )
