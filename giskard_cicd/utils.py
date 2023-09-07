import pathlib


def dump_model_and_dataset_for_cicd(artifact_path, giskard_model, giskard_dataset):
    from giskard.core.model_validation import validate_model, validate_model_loading_and_saving

    try:
        reloaded_model = validate_model_loading_and_saving(giskard_model)
    except Exception as e:
        raise Exception("An issue occured during the serialization/deserialization of your model. Please submit the traceback as a GitHub issue in the following "
                        "repository for further assistance: https://github.com/Giskard-AI/giskard.") from e
    try:
        validate_model(reloaded_model, giskard_dataset)
    except Exception as e:
        raise Exception("An issue occured during the validation of your model. Please submit the traceback as a GitHub issue in the following "
                        "repository for further assistance: https://github.com/Giskard-AI/giskard.") from e

    pathlib.Path(artifact_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(artifact_path+'/artifacts').mkdir(parents=True, exist_ok=True)
    pathlib.Path(artifact_path+'/artifacts/dataset').mkdir(parents=True, exist_ok=True)
    pathlib.Path(artifact_path+'/artifacts/model').mkdir(parents=True, exist_ok=True)

    #TODO: change the Dataset.save() method to be like Model.save(), i.e. without the id requirement
    giskard_dataset.save(pathlib.Path(artifact_path+"/artifacts/dataset"), 0)
    giskard_model.save(pathlib.Path(artifact_path+"/artifacts/model"))
    print("Your model and dataset are successfully dumped for CI/CD.")
