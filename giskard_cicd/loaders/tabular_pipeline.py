from typing import Any, Dict
from transformers import Pipeline, AutoModel, AutoConfig
import keras
import huggingface_hub
import joblib
import os
import json
from skops.io import load
from transformers.pipelines.base import GenericTensor
from transformers.utils import ModelOutput

class TabularPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        self._num_workers = 0
        # get model parameter from args
        self.model_id = kwargs.pop("model", None)
        self.model_dir = huggingface_hub.snapshot_download(self.model_id)
        serialization = kwargs.pop("serialization", None)
        for f in os.listdir(self.model_dir):
            if serialization == "skops" and ".pkl" in f:
                self.model = load(self.model_dir + "/" + f)
            if ".joblib" in f: # joblib
                self.model = joblib.load(self.model_dir + "/" + f)
            if "config.json" in f:
                config_file = json.load(open(self.model_dir + "/" + f))
                if "sklearn" in config_file.keys():
                    self.config = config_file["sklearn"]
                    if "columns" in self.config.keys():
                        self.config["features"] = self.config["columns"]
                else:
                    self.config = config_file
            if ".pt" in f: # pytorch
                self.model = AutoModel.from_pretrained(self.model_dir)
                if "config.json" in f:
                    self.model.config = AutoConfig.from_pretrained(self.model_dir)
            if "model.pb" in f: # keras
                self.model = keras.models.load_model(self.model_dir)
            if "modelRun.json" in f:
                raise ValueError(
                    "MLConsole models are not suppoerted."
                )
    
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)
        
    def _sanitize_parameters(self, **kwargs):
        kwargs = super()._sanitize_parameters(**kwargs)
        return kwargs
    def _forward(self, input_tensors: Dict[str, GenericTensor], **forward_parameters: Dict) -> ModelOutput:
        return super()._forward(input_tensors, **forward_parameters)
    
    def preprocess(self, *args, **kwargs):
        return super().preprocess(*args, **kwargs)
    
    def postprocess(self, model_outputs: ModelOutput, **postprocess_parameters: Dict) -> Any:
        return super().postprocess(model_outputs, **postprocess_parameters)
    