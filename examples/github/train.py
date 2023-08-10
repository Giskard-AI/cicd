import giskard
from pathlib import Path
import pickle
import cloudpickle

# Training
clf, data = giskard.demo.titanic()

# Let's say that you save your model after training
local_path = "."
try:
    model_file = Path(local_path) / "model.pkl"
    with open(model_file, "wb") as f:
        cloudpickle.dump(clf, f, protocol=pickle.DEFAULT_PROTOCOL)
except ValueError:
    raise ValueError("An error occurred during the saving of model.")
