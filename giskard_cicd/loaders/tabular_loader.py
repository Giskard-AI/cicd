from transformers import Pipeline, AutoModel
import huggingface_hub
import joblib
import os

class TabularClassificationPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        self._model_type = "tabular-classification"
        self._check_model_type(self._model_type)
        self.pipeline_tag = "tabular-classification"
        # get model parameter from args
        model_id = kwargs.pop("model", None)
        self.model_dir = huggingface_hub.snapshot_download(model_id)
        for f in os.listdir(self.model_dir):
            if ".joblib" in f:
                self.model = joblib.load(self.model_dir + "/" + f)
            if ".pb" in f:
                self.model = AutoModel.from_pretrained(self.model_dir)

    def _sanitize_parameters(self, **kwargs):
        kwargs = super()._sanitize_parameters(**kwargs)
        return kwargs
    
    def _check_model_type(self, model_type):
        if model_type != self._model_type:
            raise ValueError(
                f"Pipeline is not of type {self._model_type} but {model_type}"
            )
        
    def _forward(self, *args, **kwargs):
        return super()._forward(*args, **kwargs)
    
    def preprocess(self, *args, **kwargs):
        return super().preprocess(*args, **kwargs)
    
    def postprocess(self, *args, **kwargs):
        return super().postprocess(*args, **kwargs)
        