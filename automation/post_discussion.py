import huggingface_hub as hf_hub
def create_discussion(repo_id, model_name, hf_token, report):
    # Create a discussion
    discussion = hf_hub.create_discussion(repo_id, title=f"Report for {model_name}", token=hf_token, description=report, repo_type="space")
    return discussion
