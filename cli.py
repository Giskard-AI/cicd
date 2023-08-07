import argparse

from giskard_cicd.loaders import HuggingFaceLoader
from giskard_cicd.pipeline.runner import PipelineRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Giskard Scanner", description="Scans a model for vulnerabilities and produces a report."
    )
    parser.add_argument(
        "--loader",
        help="Which loader to use to set up the model. Currently only `huggingface` is supported.",
        required=True,
    )
    parser.add_argument("--model", help="The model to scan.", required=True)
    parser.add_argument("--dataset_id", help="The validation or test dataset that will be used.")
    parser.add_argument(
        "--dataset_split", help="The split of the dataset to use. If not provided, the best split will be selected."
    )
    parser.add_argument("--dataset_config", help="The name of the dataset config subset to use.")
    parser.add_argument("--output", help="Optional name of the output file.")

    args = parser.parse_args()

    runner = PipelineRunner(loaders={"huggingface": HuggingFaceLoader()}, detectors=["robustness"])
    report = runner.run(
        loader_id=args.loader,
        model_id=args.model,
        dataset_id=args.dataset_id,
        dataset_split=args.dataset_split,
        dataset_config=args.dataset_config,
    )

    # In the future, write markdown report or directly push to discussion.
    html_report = report.to_html()

    if args.output:
        with open(args.output, "w") as f:
            f.write(html_report)
    else:
        # To stdout
        print(html_report)
