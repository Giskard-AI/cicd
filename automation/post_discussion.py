import huggingface_hub as hf_hub
import markdown
import re
from time import sleep

def construct_opening(dataset_id, dataset_config, dataset_split, vulnerability_count):
    opening = """
    \nHey Team!🤗✨ <br />We’re thrilled to share some amazing evaluation results that’ll make your day!🎉📊<br />
    """
    opening += f"""
    \nWe have identified {vulnerability_count} potential vulnerabilities in your model based on an automated scan.
    """
    if dataset_id is not None:  
        opening += f"""
    \nThis automated analysis evaluated the model on the dataset {dataset_id} (subset `{dataset_config}`, split `{dataset_split}`).
        """
    return opening

def construct_closing():
    disclaimer = """
    \n\n**Disclaimer**: it's important to note that automated scans may produce false positives or miss certain vulnerabilities. We encourage you to review the findings and assess the impact accordingly.\n
    """
    whatsnext = """
    \n### 💡 What's Next?\n- Checkout the [Giskard Space](https://huggingface.co/spaces/giskardai/giskard) and improve your model. \n - [The Giskard community](https://github.com/Giskard-AI/giskard) is always buzzing with ideas. 🐢🤔 What do you want to see next? Your feedback is our favorite fuel, so drop your thoughts in the community forum! 🗣️💬 Together, we're building something extraordinary.\n
    """
    thanks = """
    \n### 🙌 Big Thanks!\nWe're grateful to have you on this adventure with us. 🚀🌟 Here's to more breakthroughs, laughter, and code magic! 🥂✨ Keep hugging that code and spreading the love! 💻 #Giskard #Huggingface #AISafety 🌈👏 Your enthusiasm, feedback, and contributions are what seek. 🌟 Keep being awesome!\n
    """
    return disclaimer + whatsnext + thanks


def construct_post_content(
    report, dataset_id, dataset_config, dataset_split, scan_report=None
):
    if scan_report is not None:
        vulnerability_count = len(scan_report.scan_result.issues)
    else:
        vulnerability_count = 0

    # Construct the content of the post
    opening = construct_opening(dataset_id, dataset_config, dataset_split, vulnerability_count)

    closing = construct_closing()

    content = f"{opening}{report}{closing}"
    return content


def save_post(report_path, path, dataset_id, dataset_config, dataset_split):
    f = open(report_path, "r")
    report = markdown.markdown(f.read())
    post = construct_post_content(report, dataset_id, dataset_config, dataset_split)
    with open(path, "w") as f:
        f.write(post)

'''
Separate report into comments
'''
def separate_report_by_issues(report):
    issues_titles = [
        "Robustness", 
        "Performance", 
        "Overconfidence",
        "Underconfidence",
        "Ethical",
        "Data Leakage",
        "Stochasticity",
        "Spurious Correlation",
        "Harmfulness", 
        "Stereotypes", 
        "Hallucination and Misinformation",
        "Sensitive Information Disclosure",
        "Output Formatting"]
    # r"\W(?=)
    regex = "\W(?=" + '|'.join(["<details>\n<summary>👉" + issue for issue in issues_titles]) + ")"
    sub_reports = re.split(regex, report)
    return sub_reports

'''
post each issue as a comment
'''
def post_issue_as_comment(discussion, issue, token, repo_id):
    comment = hf_hub.comment_discussion(
        repo_id=repo_id,
        repo_type="space",
        discussion_num=discussion.num,
        comment=issue,
        token=token,
    )
    return comment

def post_too_long_report_in_comments(discussion, report, token, repo_id):
    sub_reports = separate_report_by_issues(report)

    for issue in sub_reports:
        post_issue_as_comment(discussion, issue, token, repo_id)
    
    post_issue_as_comment(discussion, construct_closing(), token, repo_id)
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
):
    if len(report) > 60000:
        vulnerability_count = len(scan_report.scan_result.issues)
        opening = construct_opening(dataset_id, dataset_config, dataset_split, vulnerability_count)
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
        post_too_long_report_in_comments(discussion, report, hf_token, repo_id)
        return discussion
    
    description = construct_post_content(
        report, dataset_id, dataset_config, dataset_split, scan_report
    )

    # Create a discussion
    discussion = hf_hub.create_discussion(
        repo_id,
        title=f"Report for {model_name}",
        token=hf_token,
        description=description,
        repo_type="space",
    )

    return discussion
