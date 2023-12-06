import yaml
import giskard as gsk
import time

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
        scan_config_path = kwargs.pop("scan_config", None)
        params, detectors = None, None
        if scan_config_path is not None:
            with open(scan_config_path) as yaml_f:
                scan_config = yaml.load(yaml_f, Loader=yaml.Loader)
            params = dict(scan_config.get("configuration", None))
            detectors = list(scan_config.get("detectors", None))

        # Load the model and dataset
        start = time.time()
        gsk_model, gsk_dataset = loader.load_giskard_model_dataset(**kwargs)
        print(f"Loading took {time.time() - start} seconds.")
        
        start = time.time()
        # Run the scanner
        scan_result = gsk.scan(gsk_model, gsk_dataset, params=params, only=detectors)
        print(f"Scan took {time.time() - start} seconds.")
        # Report
        report = PipelineReport(scan_result)

        return report
