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

    command_template = Template("python cli.py --loader huggingface --model $model --dataset $dataset "
                                "--dataset_split validation --output ${output_path}/${model}__default_scan_with__${dataset}.html")

    check_exist_template = Template("${output_path}/${model}__default_scan_with__${dataset}.html")

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    for i in range(int(args.first_Nmodels)):
        row = df.iloc[i]
        model = row.modelId
        dataset = literal_eval(row.datasets)[0]

        result_path = check_exist_template.substitute(model=model, dataset=dataset, output_path=args.output_path)
        if os.path.exists(result_path):
            answer = input(f"{result_path} already exists, Overwrite[o] or Skip[s]? ")

            while answer not in ["o", "s"]:
                answer = input("Unvalid answer, please choose between 'o' and 's'")

            if answer == 'o':
                os.remove(result_path)
            elif answer == 's':
                continue

        command = command_template.substitute(model=model, dataset=dataset, output_path=args.output_path)
        os.system(command)
