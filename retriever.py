# %%

import huggingface_hub


def model_has_dataset(model):
    for tag in model.tags:
        if tag.startswith("dataset:"):
            return True
    return False


# %%

MODEL_TYPE = "text-classification"

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

df.to_parquet(f"models_{MODEL_TYPE}.parquet", index=False)

# %%
