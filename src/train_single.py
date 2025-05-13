import functools
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from utils.model_creators import create_LSTM

import json
from gossiplearning.config import Config
from utils.single_node_training import train_single_node

datasets_folder = Path("data/datasets")
N_SIMULATIONS = 10
dataset_name = "4in_notscaled"

with open("config.json", "r") as f:
    config = Config.model_validate(json.load(f))


model_creator = functools.partial(
    create_LSTM,
    config=config,
)
model_creator().summary()


def run_worker(data: tuple[str, int, int]) -> None:
    dataset, sim, node = data

    print(f"Training model: sim {sim} node {node} ")
    sys.stdout.flush()

    train_single_node(
        config=config,
        datasets_folder=datasets_folder / dataset / str(sim) / dataset_name,
        output_folder=datasets_folder / dataset / str(sim) / dataset_name / "models",
        model_creator=model_creator,
        node=node,
        verbose=1,
    )

    print(f"Finished model: sim {sim} node {node} ")
    sys.stdout.flush()


worker_jobs: list[tuple[str, int, int]] = []


if __name__ == "__main__":
    for n in range(6, 23, 2):
        if n != 10:
            for i in range(N_SIMULATIONS):
                for j in range(n):
                    run_worker((f"porto_{n}n_3k", i, j))
