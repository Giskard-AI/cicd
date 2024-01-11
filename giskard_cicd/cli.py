import argparse
import json
import logging
import pickle
import uuid

from giskard_cicd.automation import (
    commit_to_dataset,
    create_discussion,
    init_dataset_commit_scheduler,
)
from giskard_cicd.loaders import GithubLoader, HuggingFaceLoader
from giskard_cicd.pipeline.runner import PipelineRunner
from giskard_cicd.utils import giskard_hub_upload_helper

logger = logging.getLogger(__file__)


def main():
    parser = argparse.ArgumentParser(
        prog="Giskard Scanner",
        description="Scans a model for vulnerabilities and produces a report.",
    )
    parser.add_argument(
        "--loader",
        help="Which loader to use to set up the model. Currently only `github` and `huggingface` are supported.",
        required=True,
    )
    parser.add_argument("--model", help="The model to scan.", required=True)
    parser.add_argument(
        "--dataset", help="The validation or test dataset that will be used."
    )
    parser.add_argument(
        "--dataset_split",
        help="The split of the dataset to use. If not provided, the best split will be selected.",
    )
    parser.add_argument(
        "--dataset_config", help="The name of the dataset config subset to use."
    )
    parser.add_argument(
        "--feature_mapping", help="The feature mapping from dataset to model input."
    )
    parser.add_argument(
        "--label_mapping", help="The label mapping from dataset to model input."
    )
    parser.add_argument(
        "--scan_config",
        help="Path to YAML file containing the configuration of the scan.",
    )
    parser.add_argument("--output", help="Optional name of the output file.")
    parser.add_argument(
        "--output_format",
        help="Format of the report (either HTML or markdown). Default is HTML.",
    )
    parser.add_argument(
        "--output_portal",
        help="The output portal of the report (either huggingface or local directory). Default is local.",
    )
    parser.add_argument("--discussion_repo", help="The repo to push the report to.")
    parser.add_argument("--hf_token", help="The token to push the report to the repo.")

    parser.add_argument(
        "--persistent_scan", help="Persistent scan report.", type=bool, default=False
    )

    parser.add_argument(
        "--leaderboard_dataset", help="The leaderboard dataset to push the report to."
    )
    parser.add_argument(
        "--inference_type",
        help="The inference type to use. Default is `hf_inference_api`.",
    )
    parser.add_argument(
        "--inference_api_token", help="The HF token to call inference API with."
    )

    # Giskard hub upload args, set --giskard_hub_api_key to upload
    parser.add_argument(
        "--giskard_hub_url",
        help="The URL to upload the scan result.",
        type=str,
        default="https://giskardai-giskard.hf.space",
    )
    parser.add_argument(
        "--giskard_hub_project_key",
        help="The project key to upload the scan result.",
        type=str,
        default="giskard_bot_project",
    )
    parser.add_argument(
        "--giskard_hub_project",
        help="The project to upload the scan result.",
        type=str,
        default="Giskard bot Project",
    )
    parser.add_argument(
        "--giskard_hub_api_key",
        help="The API Key to upload the scan result.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--giskard_hub_hf_token",
        help="The Hugging Face Spaces token to upload the scan result to a private Space.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--giskard_hub_unlock_token",
        help="The unlock token to upload the scan result to a locked Space.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    supported_loaders = {
        "huggingface": HuggingFaceLoader(),
        "github": GithubLoader(),
    }

    runner = PipelineRunner(loaders=supported_loaders)


    runner_kwargs = {
        "loader_id": args.loader,
        "model": args.model,
        "dataset": args.dataset,
        "scan_config": args.scan_config,
    }

    if args.loader == "huggingface":
        runner_kwargs.update(
            {
                "dataset_split": args.dataset_split,
                "dataset_config": args.dataset_config,
                "hf_token": args.hf_token,
                "inference_type": args.inference_type,
                "inference_api_token": args.inference_api_token,
            }
        )
        try:
            feature_mapping = json.loads(args.feature_mapping)
        except Exception:
            feature_mapping = None
        try:
            label_mapping = json.loads(args.label_mapping)
            # Update labels to have integer index, which is not allowed in JSON
            label_mapping = {int(k): v for k, v in label_mapping.items()}
        except Exception:
            label_mapping = None
        runner_kwargs.update({"manual_feature_mapping": feature_mapping})
        runner_kwargs.update({"classification_label_mapping": label_mapping})

    # Hide critical information
    anonymous_runner_kwargs = runner_kwargs.copy()
    if "hf_token" in anonymous_runner_kwargs:
        anonymous_runner_kwargs.pop("hf_token")
    logger.info(
        f'Running scanner with {anonymous_runner_kwargs} to evaluate "{args.model}" model'
    )
    report = runner.run(**runner_kwargs)

    if args.persistent_scan:
        run_args = [
            args.model,
            args.dataset,
            args.dataset_config,
            args.dataset_split,
            args.feature_mapping,
            args.label_mapping,
        ]
        run_info = "+".join(filter(lambda x: x is not None, run_args))
        fn = f"{str(uuid.uuid5(uuid.NAMESPACE_OID, run_info))}.pkl"
        with open(fn, "wb") as f:
            pickle.dump(report, f)
        print(f"Scan report persisted in {fn}")

    if args.giskard_hub_api_key is not None:
        # Upload to a Giskard Hub instance
        logger.info(f"Uploading to {args.giskard_hub_url}")
        giskard_hub_upload_helper(
            args,
            report,
            url=args.giskard_hub_url,
            project_key=args.giskard_hub_project_key,
            project=args.giskard_hub_project,
            key=args.giskard_hub_api_key,
            hf_token=args.giskard_hub_hf_token,
            unlock_token=args.giskard_hub_unlock_token,
        )

    # In the future, write markdown report or directly push to discussion.
    if args.output_format == "markdown":
        rendered_report = report.to_markdown(template="huggingface")
    else:
        rendered_report = report.to_html()

    if args.output_portal == "huggingface":
        # Push to discussion
        # FIXME: dataset config and dataset split might have been inferred
        discussion = create_discussion(
            args.discussion_repo,
            args.model,
            args.hf_token,
            rendered_report,
            args.dataset,
            args.dataset_config,
            args.dataset_split,
            report,
        )

        if args.leaderboard_dataset:  # Commit to leaderboard dataset
            scheduler = init_dataset_commit_scheduler(
                hf_token=args.hf_token, dataset_id=args.leaderboard_dataset
            )

            try:
                commit_to_dataset(
                    scheduler,
                    args.model,
                    args.dataset,
                    args.dataset_config,
                    args.dataset_split,
                    discussion,
                    report,
                )
            except Exception as e:
                logging.debug(f"Failed to commit to dataset: {e}")
 

    if args.output:
        with open(args.output, "w") as f:
            f.write(rendered_report)
    elif args.output_format == "markdown":
        # To stdout
        print(rendered_report)
        model_name = args.model.split("/")[-1]
        with open(f"{model_name}_report.md", "w") as f:
            f.write(rendered_report)
    else:
        model_name = args.model.split("/")[-1]
        with open(f"{model_name}_report.html", "w") as f:
            f.write(rendered_report)
