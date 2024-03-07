from typing import Optional
import huggingface_hub as hf_hub
import markdown
from time import sleep
import logging

logger = logging.getLogger(__file__)
GISKARD_HUB_URL = "https://huggingface.co/spaces/giskardai/giskard"


def construct_opening(
    dataset_id, dataset_config, dataset_split, vulnerability_count, persistent_url
):
    opening = """
    \nHi Team,\n\nThis is a report from <b>Giskard Bot Scan 🐢</b>.<br />
    """
    if vulnerability_count == 0:
        opening += """
        \nWe have not identified any potential vulnerabilities in your model based on an automated scan.
        """
    else:
        opening += f"""
        \nWe have identified {vulnerability_count} potential vulnerabilities in your model based on an automated scan.
        """
    if dataset_id is not None:
        opening += f"""
        \nThis automated analysis evaluated the model on the dataset {dataset_id} (subset `{dataset_config}`, split `{dataset_split}`).
        """
    if persistent_url is not None:
        opening += f"""
        \nYou can find a full version of scan report [here]({persistent_url}).
        """
    return opening


def construct_closing(test_suite_url=None):
    giskard_hub_wording = f"""
    \n\nWe've generated test suites according to your scan results! Checkout the [Test Suite in our Giskard Space]({test_suite_url}) and [Giskard Documentation](https://docs.giskard.ai/en/stable/getting_started/quickstart/quickstart_nlp.html) to learn more about how to test your model.
    """

    if test_suite_url is None:
        giskard_hub_wording = f"""
        \n\nCheckout out the [Giskard Space]({GISKARD_HUB_URL}) and [Giskard Documentation](https://docs.giskard.ai/en/stable/getting_started/quickstart/quickstart_nlp.html) to learn more about how to test your model.
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
    persistent_url=None,
):
    if scan_report is not None:
        vulnerability_count = len(scan_report.scan_result.issues)
    else:
        vulnerability_count = 0

    # Construct the content of the post
    opening = construct_opening(
        dataset_id,
        dataset_config,
        dataset_split,
        vulnerability_count,
        persistent_url,
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


class Issue:
    def __init__(self, description, examples):
        self.description = description
        self.examples = examples

    def __len__(self):
        return len(self.description) + len(self.examples)

    def trim_examples(self):
        # get characters count of the examples
        if len(self.examples) > 60000:
            self.examples = "examples are too long to be displayed"


def load_report_to_issues(report):
    splited_issues = []
    # <!-- issue --> is used to separate the issues
    issues = [issue for issue in report.split("<!-- issue -->") if len(issue) > 0]
    # <!-- examples --> is used to separate the examples
    for issue in issues:
        descriptions = []
        examples = []
        splited_issue = issue.split("<!-- examples -->")
        descriptions.append(splited_issue[0])
        for sub_issue in splited_issue[1:]:
            res = sub_issue.split("</details>")
            for i in range(0, len(res), 2):
                if len(res[i]) == 0 or len(set(res[i])) < 10:
                    continue
                examples.append(res[i])
                if i + 1 < len(res):
                    descriptions.append(res[i + 1])
        splited_issues.extend(
            [
                Issue(description, example)
                for description, example in zip(descriptions, examples)
            ]
        )
    return splited_issues


def post_issue_as_comment(discussion, issue, token, repo_id, repo_type):
    try:
        comment = issue
        if isinstance(issue, Issue):
            comment = issue.description + issue.examples
        hf_hub.comment_discussion(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_num=discussion.num,
            comment=comment,
            token=token,
        )
    except Exception as e:
        logger.debug(f"Failed to post issue as comment: {e}")


def post_too_long_report_in_comments(
    discussion, report, token, repo_id, repo_type, test_suite_url=None
):
    issues = load_report_to_issues(report)
    for issue in issues:
        if len(issue) > 60000:
            issue.trim_examples()
        post_issue_as_comment(discussion, issue, token, repo_id, repo_type)
        # To keep order of sub-posts
        sleep(1)
    post_issue_as_comment(
        discussion, construct_closing(test_suite_url), token, repo_id, repo_type
    )
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
    persistent_url: Optional[str] = None,
):
    # Detect repo type
    repo_type = "space"
    if repo_id is None:
        repo_type = "model"
        repo_id = model_name

    if len(report) > 60000:
        vulnerability_count = len(scan_report.scan_result.issues)
        opening = construct_opening(
            dataset_id,
            dataset_config,
            dataset_split,
            vulnerability_count,
            persistent_url,
        )
        discussion = hf_hub.create_discussion(
            repo_id,
            title=f"Report for {model_name}",
            token=hf_token,
            description=opening,
            repo_type=repo_type,
        )
        # wait for the discussion to be created
        # otherwise, the comments will be posted before the discussion description
        sleep(1)
        post_too_long_report_in_comments(
            discussion,
            report,
            hf_token,
            repo_id,
            repo_type,
            test_suite_url=test_suite_url,
        )
        return discussion

    description = construct_post_content(
        report,
        dataset_id,
        dataset_config,
        dataset_split,
        scan_report,
        test_suite_url=test_suite_url,
        persistent_url=persistent_url,
    )

    discussion = hf_hub.create_discussion(
        repo_id,
        title=f"Report for {model_name}",
        token=hf_token,
        description=description,
        repo_type=repo_type,
    )

    return discussion
