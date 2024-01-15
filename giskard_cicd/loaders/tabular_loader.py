from transformers import Pipeline

class TabularClassificationPipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model_type = "tabular-classification"
        self._check_model_type(self._model_type)
        self.pipeline_tag = "tabular-classification"

    def _sanitize_parameters(self, **kwargs):
        kwargs = super()._sanitize_parameters(**kwargs)
        return kwargs
    
    def _check_model_type(self, model_type):
        if model_type != self._model_type:
            raise ValueError(
                f"Pipeline is not of type {self._model_type} but {model_type}"
            )
        
    def _forward(self, *args, **kwargs):
        raise NotImplementedError("Not implemented yet")
    
    def preporocess(self, *args, **kwargs):
        return super().preprocess(*args, **kwargs)
    
    def postprocess(self, *args, **kwargs):
        return super().postprocess(*args, **kwargs)