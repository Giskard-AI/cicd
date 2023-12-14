import huggingface_hub as hf_hub
import markdown


def construct_post_content(
    report, dataset_id, dataset_config, dataset_split, scan_report
):
    vulnerability_count = len(scan_report.scan_result.issues)

    # Construct the content of the post
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
    disclaimer = """
    \n\n**Disclaimer**: it's important to note that automated scans may produce false positives or miss certain vulnerabilities. We encourage you to review the findings and assess the impact accordingly.\n
    """
    whatsnext = """
    \n### 💡 What's Next?\n- Checkout the [Giskard Space](https://huggingface.co/spaces/giskardai/giskard) and improve your model. \n - [The Giskard community](https://github.com/Giskard-AI/giskard) is always buzzing with ideas. 🐢🤔 What do you want to see next? Your feedback is our favorite fuel, so drop your thoughts in the community forum! 🗣️💬 Together, we're building something extraordinary.\n
    """
    thanks = """
    \n### 🙌 Big Thanks!\nWe're grateful to have you on this adventure with us. 🚀🌟 Here's to more breakthroughs, laughter, and code magic! 🥂✨ Keep hugging that code and spreading the love! 💻 #Giskard #Huggingface #AISafety 🌈👏 Your enthusiasm, feedback, and contributions are what seek. 🌟 Keep being awesome!\n
    """

    content = f"{opening}{report}{disclaimer}{whatsnext}{thanks}"
    return content


def save_post(report_path, path, dataset_id, dataset_config, dataset_split):
    f = open(report_path, "r")
    report = markdown.markdown(f.read())
    post = construct_post_content(report, dataset_id, dataset_config, dataset_split)
    with open(path, "w") as f:
        f.write(post)


def create_discussion(
    repo_id,
    model_name,
    hf_token,
    report,
    dataset_id,
    dataset_config,
    dataset_split,
    scan_report,
):
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
