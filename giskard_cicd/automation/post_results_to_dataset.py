from huggingface_hub import CommitScheduler
from .utils import DATASET_ID, ISSUE_GROUPS
from uuid import uuid4
from pathlib import Path
from datetime import datetime
import json
import logging
from collections import defaultdict

RESULT_DIR = Path("json_dataset")


def init_dataset_commit_scheduler(hf_token=None, dataset_id=None):
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        scheduler = CommitScheduler(
            repo_id=DATASET_ID or dataset_id,
            repo_type="dataset",
            folder_path=RESULT_DIR,
            token=hf_token,
            path_in_repo=".",
        )

        return scheduler
    except Exception as e:
        logging.debug(f"Failed to initialize dataset commit scheduler: {e}")
        return None


def commit_to_dataset(
    scheduler,
    model_name,
    dataset_id,
    dataset_config,
    dataset_split,
    discussion,
    scan_report: object,
):
    if not scheduler:
        raise ValueError("Scheduler is not initialized")
    
    new_record = dict.fromkeys(ISSUE_GROUPS, 0)

    new_record.update(
        {
            "task": "text_classification",
            "model_id": model_name,
            "dataset_id": dataset_id,
            "dataset_config": dataset_config,
            "dataset_split": dataset_split,
            "total_issues": len(scan_report.scan_result.issues),
            "report_link": discussion.url,
            "timestamp": datetime.now().isoformat(),
        }
    )

    if len(scan_report.scan_result.issues) != 0:
        issues_by_group = defaultdict(list)
        for issue in scan_report.scan_result.issues:
            issues_by_group[issue.group].append(issue)

        for group, issues in issues_by_group.items():
            new_record[group.name] = len(issues)

    RESULT_PATH = scheduler.folder_path / f"train-{uuid4()}.json"
    with scheduler.lock:
        with RESULT_PATH.open("a") as f:
            json.dump(new_record, f)
            f.write("\n")
