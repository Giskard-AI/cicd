import giskard as gsk


class PipelineReport:
    def __init__(self, scan_result):
        self.scan_result = scan_result

    def to_html(self):
        return self.scan_result.to_html()

    def to_markdown(self, template):
        return self.scan_result.to_markdown(template="github")


class PipelineRunner:
    def __init__(self, loaders, detectors):
        self.loaders = loaders
        self.detectors = detectors

    def run(self, loader_id, **kwargs):

        # Get the loader
        loader = self.loaders[loader_id]

        # Load the model and dataset
        gsk_model, gsk_dataset = loader.load_giskard_model_dataset(**kwargs)

        # Run the scanner
        scan_result = gsk.scan(gsk_model, gsk_dataset, only=self.detectors)

        # Report
        report = PipelineReport(scan_result)

        return report
