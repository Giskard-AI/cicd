from json import JSONDecodeError
import logging
import pathlib

from giskard import GiskardClient
from giskard.client.giskard_client import GiskardError

logger = logging.getLogger(__file__)

GISKARD_HUB_UNLOCK_STATUS_ENDPOINT = "hfs/unlock"


def dump_model_and_dataset_for_cicd(artifact_path, giskard_model, giskard_dataset):
    from giskard.core.model_validation import (
        validate_model,
        validate_model_loading_and_saving,
    )

    try:
        reloaded_model = validate_model_loading_and_saving(giskard_model)
    except Exception as e:
        raise Exception(
            "An issue occured during the serialization/deserialization of your model. Please submit the traceback as a GitHub issue in the following "
            "repository for further assistance: https://github.com/Giskard-AI/giskard."
        ) from e
    try:
        validate_model(reloaded_model, giskard_dataset)
    except Exception as e:
        raise Exception(
            "An issue occured during the validation of your model. Please submit the traceback as a GitHub issue in the following "
            "repository for further assistance: https://github.com/Giskard-AI/giskard."
        ) from e

    pathlib.Path(artifact_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(artifact_path + "/artifacts").mkdir(parents=True, exist_ok=True)
    pathlib.Path(artifact_path + "/artifacts/dataset").mkdir(
        parents=True, exist_ok=True
    )
    pathlib.Path(artifact_path + "/artifacts/model").mkdir(parents=True, exist_ok=True)

    # TODO: change the Dataset.save() method to be like Model.save(), i.e. without the id requirement
    giskard_dataset.save(pathlib.Path(artifact_path + "/artifacts/dataset"), 0)
    giskard_model.save(pathlib.Path(artifact_path + "/artifacts/model"))
    logger.info("Your model and dataset are successfully dumped for CI/CD.")


def giskard_hub_upload_helper(
    args, report, url, project_key, project, key, hf_token=None, unlock_token=None
):
    need_relock = False
    try:
        client = GiskardClient(
            url=url,
            key=key,
            hf_token=hf_token,
        )

        # Check unlock state
        unlock_resp = client.session.get(GISKARD_HUB_UNLOCK_STATUS_ENDPOINT).json()

        if not unlock_resp["unlocked"] and unlock_token is None:
            logger.warn("Cannot upload to a locked Giskard Hub without unlock token")
            return
        elif not unlock_resp["unlocked"]:
            resp = client.session.post(
                GISKARD_HUB_UNLOCK_STATUS_ENDPOINT,
                json={
                    "token": unlock_token,
                    "unlocked": True,
                },
            ).json()
            if not resp["unlocked"]:
                # Unlock failed
                logger.warn(
                    "Cannot upload to a locked Giskard Hub due to wrong unlock token"
                )
                return
            # Unlock succeed: remeber to lock it
            need_relock = resp["unlocked"]

        # Create project
        project_name = project if project is not None else "Giskard bot Project"
        project_key = (
            project_key
            if project_key is not None
            else project_name.lower().replace(" ", "_")
        )
        try:
            client.create_project(
                project_key=project_key,
                name=project_name,
                description="Project with scanned models and datasets from Hugging Face",
            )
        except GiskardError:
            logger.info(f"Project {project_key} already exists.")

        suite = report.scan_result.generate_test_suite(
            f'Test suite for "{args.model}" model, by Giskard bot'
        )
        suite.upload(client, project_key)
    except GiskardError as e:
        logger.warn(f"Uploading project failed: {e}")
        return
    except JSONDecodeError as e:
        logger.warn(f"Uploading project failed due to response decoding: {e}")
        return

    finally:
        if need_relock:
            logger.debug("Relocking the Space")
            client.session.post(
                GISKARD_HUB_UNLOCK_STATUS_ENDPOINT,
                json={
                    "token": unlock_token,
                    "unlocked": False,
                },
            )
            logger.debug("Space is relocked")
