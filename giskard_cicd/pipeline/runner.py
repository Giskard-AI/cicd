import yaml
import giskard as gsk
import time

import logging

logger = logging.getLogger(__file__)


class PipelineReport:
    def __init__(self, scan_result):
        self.scan_result = scan_result

    def to_html(self):
        return self.scan_result.to_html()

    def to_markdown(self, template):
        return self.scan_result.to_markdown(template=template)

    def to_avid(self):
        return self.scan_result.to_avid()


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

            params = scan_config.get("configuration")
            detectors = scan_config.get("detectors")

        start = time.time()
        # Load the model and dataset
        gsk_model, gsk_dataset = loader.load_giskard_model_dataset(**kwargs)
        logger.info(f"Loading took {time.time() - start:.2f}s")

        start = time.time()
        # Run the scanner
        scan_result = gsk.scan(gsk_model, gsk_dataset, params=params, only=detectors)
        logger.info(f"Scanning took {time.time() - start:.2f}s")

        # Report
        report = PipelineReport(scan_result)

        return report
