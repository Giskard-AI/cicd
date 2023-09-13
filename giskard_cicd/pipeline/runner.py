import yaml
import giskard as gsk


class PipelineReport:
    def __init__(self, scan_result):
        self.scan_result = scan_result

    def to_html(self):
        return self.scan_result.to_html()

    def to_markdown(self, template):
        return self.scan_result.to_markdown(template="github")


class PipelineRunner:
    def __init__(self, loaders):
        self.loaders = loaders

    def run(self, loader_id, **kwargs):

        # Get the loader
        loader = self.loaders[loader_id]

        # Get scan configuration
        if kwargs["scan_config"] is not None:
            with open(kwargs["scan_config"]) as yaml_f:
                scan_config = yaml.load(yaml_f, Loader=yaml.Loader)
            params = dict(scan_config.get("configuration", None))
            detectors = list(scan_config.get("detectors", None))
        kwargs.pop("scan_config")

        # Load the model and dataset
        gsk_model, gsk_dataset = loader.load_giskard_model_dataset(**kwargs)

        # Run the scanner
        scan_result = gsk.scan(gsk_model, gsk_dataset, params=params, only=detectors)

        # Report
        report = PipelineReport(scan_result)

        return report
