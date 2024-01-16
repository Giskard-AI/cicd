from .tabular_pipeline import TabularPipeline

class TabularRegressionPipeline(TabularPipeline):
    def __init__(self, *args, **kwargs):
        self._model_type = "regression"
        self._check_model_type(self._model_type)
        self.pipeline_tag = "tabular-regression"
        # get model parameter from args
        super().__init__(*args, **kwargs)

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
        