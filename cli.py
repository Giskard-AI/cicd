import argparse
import sys

from giskard_cicd.loaders import HuggingFaceLoader, BaseLoader
from giskard_cicd.pipeline.runner import PipelineRunner


# TODO: simplify for find a more robust logic
def get_custom_loader_class(loader_path):
    import importlib
    import inspect
    module_name = args.loader_module_path.split(".py")[0].split("/")[-1]
    spec = importlib.util.spec_from_file_location(module_name, args.loader_module_path)
    loader_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = loader_module
    spec.loader.exec_module(loader_module)
    for cls_name, cls in inspect.getmembers(sys.modules[module_name]):
        if inspect.isclass(cls) and issubclass(cls, BaseLoader) and cls != BaseLoader:
            return cls


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Giskard Scanner", description="Scans a model for vulnerabilities and produces a report."
    )
    parser.add_argument(
        "--loader",
        help="Which loader to use to set up the model. Currently only `huggingface` and `github` are supported.",
        required=True,
    )
    parser.add_argument(
        "--loader_module_path",
        help="The path to the module containing the loader class to use in case of a `github` loader.",
    )
    parser.add_argument("--model", help="The model to scan.", required=True)
    parser.add_argument("--dataset", help="The validation or test dataset that will be used.")
    parser.add_argument(
        "--dataset_split", help="The split of the dataset to use. If not provided, the best split will be selected."
    )
    parser.add_argument("--dataset_config", help="The name of the dataset config subset to use.")
    parser.add_argument("--output", help="Optional name of the output file.")

    args = parser.parse_args()

    loaders = dict()
    config = {"loader_id": args.loader,
              "model": args.model,
              "dataset": args.dataset}
    if args.loader == "huggingface":
        loaders.update({"huggingface": HuggingFaceLoader()})
        # huggingface specific args
        config.updated({
            "dataset_split": args.dataset_split,
            "dataset_config": args.dataset_config})

    elif args.loader == "github":
        if not hasattr(args, "loader_module_path"):
            raise Exception("the argument 'loader_module_path' should be provided "
                            "when the 'loader' chosen is 'github'.")

        loader_class = get_custom_loader_class(args.loader_module_path)
        loaders.update({"github": loader_class()})
    else:
        raise ValueError("Currently only `huggingface` and `github` loaders are supported.")

    runner = PipelineRunner(loaders=loaders, detectors=["robustness"])
    report = runner.run(**config)

    # In the future, write markdown report or directly push to discussion.
    html_report = report.to_html()

    if args.output:
        with open(args.output, "w") as f:
            f.write(html_report)
    else:
        # To stdout
        print(html_report)
