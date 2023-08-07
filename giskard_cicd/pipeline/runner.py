import giskard as gsk


class PipelineReport:
    def __init__(self, scan_result):
        self.scan_result = scan_result

    def to_html(self):
        return self.scan_result.to_html()


class PipelineRunner:
    def __init__(self, loaders, detectors):
        self.loaders = loaders
        self.detectors = detectors

    def run(self, loader_id, model_id, **args):
        # Get the loader
        loader = self.loaders[loader_id]

        # Load the model and dataset
        gsk_model, gsk_dataset = loader.load_model_dataset(model_id, **args)

        # Run the scanner
        scan_result = gsk.scan(gsk_model, gsk_dataset, only=self.detectors)

        # Report
        report = PipelineReport(scan_result)

        return report
