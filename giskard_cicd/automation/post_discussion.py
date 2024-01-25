from typing import Optional
import huggingface_hub as hf_hub
import markdown
import re
from time import sleep
from .utils import ISSUE_GROUPS

GISKARD_HUB_URL = "https://huggingface.co/spaces/giskardai/giskard"

def construct_opening(dataset_id, dataset_config, dataset_split, vulnerability_count):
    opening = """
    \nThis is a report from <b>Giskard Scan 🐢</b>.<br />
    """
    opening += f"""
    \nWe have identified {vulnerability_count} potential vulnerabilities in your model based on an automated scan.
    """
    if dataset_id is not None:
        opening += f"""
        \nThis automated analysis evaluated the model on the dataset {dataset_id} (subset `{dataset_config}`, split `{dataset_split}`).
        """
    return opening


def construct_closing(test_suite_url=None):
    giskard_hub_wording = f"""
    \n\nWe've generated test suites according to your scan results! Checkout the [Test Suite in our Giskard Space]({test_suite_url})
    """
    
    if test_suite_url is None:
        giskard_hub_wording = f"""
        \n\nCheckout out the [Giskard Space]({GISKARD_HUB_URL}) and improve your model.
        """
    
    disclaimer = """
    \n\n**Disclaimer**: it's important to note that automated scans may produce false positives or miss certain vulnerabilities. We encourage you to review the findings and assess the impact accordingly.\n
    """
    return giskard_hub_wording + disclaimer


def construct_post_content(
    report,
    dataset_id,
    dataset_config,
    dataset_split,
    scan_report=None,
    test_suite_url=None,
):
    if scan_report is not None:
        vulnerability_count = len(scan_report.scan_result.issues)
    else:
        vulnerability_count = 0

    # Construct the content of the post
    opening = construct_opening(
        dataset_id, dataset_config, dataset_split, vulnerability_count
    )

    closing = construct_closing(test_suite_url)

    content = f"{opening}{report}{closing}"
    return content


def save_post(report_path, path, dataset_id, dataset_config, dataset_split):
    f = open(report_path, "r")
    report = markdown.markdown(f.read())
    post = construct_post_content(report, dataset_id, dataset_config, dataset_split)
    with open(path, "w") as f:
        f.write(post)


def separate_report_by_issues(report):
    # TODO: add markdown comments to the report as a split marker
    regex = (
        "\W(?="
        + "|".join(["<details>\n<summary>👉" + issue for issue in ISSUE_GROUPS])
        + ")"
    )
    sub_reports = re.split(regex, report)
    return sub_reports


def post_issue_as_comment(discussion, issue, token, repo_id):
    comment = hf_hub.comment_discussion(
        repo_id=repo_id,
        repo_type="space",
        discussion_num=discussion.num,
        comment=issue,
        token=token,
    )
    return comment


def post_too_long_report_in_comments(
    discussion, report, token, repo_id, test_suite_url=None
):
    sub_reports = separate_report_by_issues(report)

    for issue in sub_reports:
        post_issue_as_comment(discussion, issue, token, repo_id)

    post_issue_as_comment(discussion, construct_closing(test_suite_url), token, repo_id)
    return discussion


def create_discussion(
    repo_id,
    model_name,
    hf_token,
    report: str,
    dataset_id,
    dataset_config,
    dataset_split,
    scan_report: object,
    test_suite_url: Optional[str],
):
    if len(report) > 60000:
        vulnerability_count = len(scan_report.scan_result.issues)
        opening = construct_opening(
            dataset_id, dataset_config, dataset_split, vulnerability_count
        )
        discussion = hf_hub.create_discussion(
            repo_id,
            title=f"Report for {model_name}",
            token=hf_token,
            description=opening,
            repo_type="space",
        )
        # wait for the discussion to be created
        # otherwise, the comments will be posted before the discussion description
        sleep(1)
        post_too_long_report_in_comments(
            discussion, report, hf_token, repo_id, test_suite_url=test_suite_url
        )
        return discussion

    description = construct_post_content(
        report,
        dataset_id,
        dataset_config,
        dataset_split,
        scan_report,
        test_suite_url=test_suite_url,
    )

    discussion = hf_hub.create_discussion(
        repo_id,
        title=f"Report for {model_name}",
        token=hf_token,
        description=description,
        repo_type="space",
    )

    return discussion
