from huggingface_hub import logging, login
import os

HF_WRITE_TOKEN = os.environ.get("HF_WRITE_TOKEN")
DATASET_ID = os.environ.get("DATASET_ID")


def check_env_vars_and_login(hf_token=None, write_permission=False):
    if HF_WRITE_TOKEN is None and hf_token is None:
        raise ValueError("Neither hf_token nor HF_WRITE_TOKEN is set")

    login(token=HF_WRITE_TOKEN or hf_token, write_permission=write_permission)


def check_env_vars_and_login_for_dataset(hf_token=None, dataset_id=None):
    if HF_WRITE_TOKEN is None and hf_token is None:
        raise ValueError("HF_WRITE_TOKEN is not set")

    # check the leaderboard dataset exists
    if DATASET_ID is None and dataset_id is None:
        raise ValueError("DATASET_ID is not set")

    check_env_vars_and_login(hf_token=hf_token, write_permission=True)

    logging.set_verbosity_debug()


ISSUE_GROUPS = [
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
    "Output Formatting",
]
