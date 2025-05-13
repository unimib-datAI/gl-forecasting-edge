import functools
import os

from gossiplearning.config import Config
from utils.model_creators import create_LSTM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

from utils.centralized_training import train_centralized_model

with open("config.json", "r") as f:
    config = Config.model_validate(json.load(f))

datasets_folder = Path("data/datasets")
N_SIMULATIONS = 10
dataset_name = "4in_notscaled"


if __name__ == "__main__":
    for n in range(6, 23, 2):
        dataset = f"porto_{n}n_3k"
        if n != 10:
            for i in range(N_SIMULATIONS):
                config.n_nodes = n
                train_centralized_model(
                    node_datasets_folder=datasets_folder
                    / dataset
                    / str(i)
                    / dataset_name,
                    model_output_path=datasets_folder
                    / dataset
                    / str(i)
                    / dataset_name
                    / "models",
                    config=config,
                    model_creator=functools.partial(create_LSTM, config=config),
                    verbose=1,
                )
