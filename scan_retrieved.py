import argparse
import pandas as pd
from ast import literal_eval
from string import Template
import os


def model_has_dataset(model):
    for tag in model.tags:
        if tag.startswith("dataset:"):
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Giskard Batch Scanner", description="Scan Retrieved HF models."
    )
    parser.add_argument(
        "--data_path",
        help="Path to retrieved models in csv format (need to run retrieve.py first).",
        required=True,
    )
    parser.add_argument("--first_Nmodels",
                        help="Number of models to be scanned from the sorted list of models available.",
                        required=True)
    parser.add_argument("--output_path",
                        help="Path of dir to save all the reports",
                        required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    df_to_be_skipped = None
    to_be_skipped_file_path = ".models_and_datasets_to_be_skipped.csv"
    if os.path.exists(to_be_skipped_file_path):
        df_to_be_skipped = pd.read_csv(to_be_skipped_file_path)

    command_template = Template("python cli.py --loader huggingface --model $model --dataset $dataset "
                                "--dataset_split $dataset_split --dataset_config $dataset_config "
                                "--output ${output_path}/${model_name}__default_scan_with__${dataset_name}.html")

    result_path_template = Template("${output_path}/${model_name}__default_scan_with__${dataset_name}.${suffix}")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    dataset_split_exceptions = {"facebook/bart-large-mnli": "validation_matched"}

    dataset_config_exceptions = {"tweet_eval": "sentiment"}

    for i in range(int(args.first_Nmodels)):
        row = df.iloc[i]
        model = row.modelId
        dataset = literal_eval(row.datasets)[0]

        message = f"{model} with {dataset}"

        if ((df_to_be_skipped['model'] == model) & (df_to_be_skipped['dataset'] == dataset)).any() \
                and df_to_be_skipped is not None:
            print(f"[{i}] ==== ⏩ skipping {message} ====")
            continue

        print(f"[{i}] ==== 🔍 scanning {message} ====")

        result_path = result_path_template.substitute(model_name=model.replace("/", "--"),
                                                      dataset_name=dataset.replace("/", "--"),
                                                      output_path=args.output_path,
                                                      suffix="html")
        if os.path.exists(result_path):
            answer = input(f"{result_path} already exists, Overwrite[o] or Skip[s]? ")

            while answer not in ["o", "s"]:
                answer = input("Invalid answer, please choose between 'o' and 's'")

            if answer == 'o':
                os.remove(result_path)
            elif answer == 's':
                continue

        command = command_template.substitute(model=model, dataset=dataset,
                                              dataset_split=dataset_split_exceptions.get(model, "validation"),
                                              dataset_config=dataset_config_exceptions.get(dataset, None),
                                              model_name=model.replace("/", "--"),
                                              dataset_name=dataset.replace("/", "--"),
                                              output_path=args.output_path)

        try:
            os.system(command)  # call the cli script in order for try, except to work
            new_row = pd.DataFrame({"model": model, "dataset": dataset, "status": "done"}, index=[0])
            df_to_be_skipped = pd.concat([df_to_be_skipped, new_row], ignore_index=True)
            df_to_be_skipped.to_csv(to_be_skipped_file_path, index=False)
        except Exception as e:
            new_row = pd.DataFrame({"model": model, "dataset": dataset, "status": "error"}, index=[0])
            df_to_be_skipped = pd.concat([df_to_be_skipped, new_row], ignore_index=True)
            df_to_be_skipped.to_csv(to_be_skipped_file_path, index=False)
            result_path = result_path_template.substitute(model_name=model.replace("/", "--"),
                                                          dataset_name=dataset.replace("/", "--"),
                                                          output_path=args.output_path,
                                                          suffix="error")
            with open(result_path, "w") as error_log:
                error_log.write(e)
            print(
                f"Something went wrong while {message}, error is logged at {result_path}. "
                "continuing with the next model...")
            # raise Exception(f"Something went wrong while {message}") from e
