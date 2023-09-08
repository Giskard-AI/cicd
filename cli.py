import argparse

from giskard_cicd.loaders import GithubLoader, HuggingFaceLoader
from giskard_cicd.pipeline.runner import PipelineRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Giskard Scanner", description="Scans a model for vulnerabilities and produces a report."
    )
    parser.add_argument(
        "--loader",
        help="Which loader to use to set up the model. Currently only `github` and `huggingface` are supported.",
        required=True,
    )
    parser.add_argument("--model", help="The model to scan.", required=True)
    parser.add_argument("--dataset", help="The validation or test dataset that will be used.")
    parser.add_argument(
        "--dataset_split", help="The split of the dataset to use. If not provided, the best split will be selected."
    )
    parser.add_argument("--dataset_config", help="The name of the dataset config subset to use.")
    parser.add_argument("--output", help="Optional name of the output file.")
    parser.add_argument("--output_format", help="Format of the report (either HTML or markdown). Default is HTML.")

    args = parser.parse_args()

    supported_loaders = {
        "huggingface": HuggingFaceLoader(),
        "github": GithubLoader(),
    }

    runner = PipelineRunner(loaders=supported_loaders, detectors=["robustness"])

    runner_kwargs = {"loader_id": args.loader,
                     "model": args.model,
                     "dataset": args.dataset}

    if args.loader == "huggingface":
        runner_kwargs.update({"dataset_split": args.dataset_split,
                              "dataset_config": args.dataset_config})

    report = runner.run(**runner_kwargs)

    # In the future, write markdown report or directly push to discussion.
    if args.output_format == "markdown":
        rendered_report = report.to_markdown(template="github")
    else:
        rendered_report = report.to_html()

    if args.output:
        with open(args.output, "w") as f:
            f.write(rendered_report)
    else:
        # To stdout
        print(rendered_report)
