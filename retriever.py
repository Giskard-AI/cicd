import argparse
import huggingface_hub


def model_has_dataset(model):
    for tag in model.tags:
        if tag.startswith("dataset:"):
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Giskard Retriever", description="Retrieves HF models that are bound to datasets."
    )
    parser.add_argument(
        "--model_type",
        help="Hugging Face model types. default: text-classification",
        required=False,
    )
    parser.add_argument("--output_format",
                        help="Format of the information retrieved. Default: parquet. Options: parquet, csv, json.")

    args = parser.parse_args()

    MODEL_TYPE = args.model_type if args.model_type is not None else "text-classification"

    models_with_dataset = filter(
        model_has_dataset, huggingface_hub.list_models(filter=MODEL_TYPE, sort="likes", direction=-1)
    )

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "modelId": m.modelId,
                "modelType": MODEL_TYPE,
                "author": m.author,
                "downloads": m.downloads,
                "likes": m.likes,
                "datasets": [t[8:] for t in m.tags if t.startswith("dataset:")],
            }
            for m in models_with_dataset
        ]
    )

    output_format = args.output_format

    if output_format is None or output_format == "parquet":
        df.to_parquet(f"models_{MODEL_TYPE}.parquet", index=False)
    elif output_format == "csv":
        df.to_csv(f"models_{MODEL_TYPE}.csv", columns=df.columns, index=False)
    elif output_format == "json":
        df.to_json(f"models_{MODEL_TYPE}.json", index=False)
