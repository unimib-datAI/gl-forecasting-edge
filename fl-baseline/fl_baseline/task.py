"""fl-baseline: A Flower / TensorFlow app."""

import datetime
import json
import os
from pathlib import Path
from typing import TypedDict

from keras.layers import Dropout
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from flwr.common import Context
from flwr.common.typing import UserConfig


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

class Dataset(TypedDict):
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray


def load_model(config: UserConfig) -> Model:
    optz = Adam(learning_rate=0.001, epsilon=1e-6)
    input_timesteps = config.get("input-steps")
    inputs = Input(shape=(input_timesteps, config.get("input-features")))

    lstm_layers = Sequential(
        [
            LSTM(
                50,
                activation="tanh",
                return_sequences=True,
            ),
            LSTM(
                50,
                activation="tanh",
                return_sequences=False,
            ),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
        ]
    )(inputs)

    outputs = [
        Dense(1, activation="relu", name=f"fn_{i}")(lstm_layers)
        for i in range(config.get("output-vars"))
    ]

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optz,
        loss={f"fn_{i}": "mse" for i in range(config.get("output-vars"))},
        metrics=["mae", "msle", "mse", "mape", RootMeanSquaredError()],
    )

    return model

def get_node_dataset(dataset_dir: str, node_index: int) -> Dataset:
    dataset: Dataset = np.load(f"{dataset_dir}/node_{node_index}.npz")
    return {
        "X_train": dataset["X_train"],
        "Y_train": dataset["Y_train"],
        "X_val": dataset["X_val"],
        "Y_val": dataset["Y_val"],
        "X_test": dataset["X_test"],
        "Y_test": dataset["Y_test"],
    }

def get_common_test_set(dataset_dir: str, n_nodes: int, perc: float) -> tuple[np.ndarray, np.ndarray]:
    datasets = [get_node_dataset(dataset_dir, n) for n in range(n_nodes)]

    test_sets = [(ds["X_test"], ds["Y_test"]) for ds in datasets]

    test: list[tuple[np.ndarray, np.ndarray]] = []

    for ts in test_sets:
        num_records = len(ts[0])
        indices = np.random.choice(
            num_records, replace=False, size=round(perc * num_records)
        )

        test.append((ts[0][indices], ts[1][indices]))

    testX = np.concatenate([t[0] for t in test])
    testY = np.concatenate([t[1] for t in test])

    return (testX, testY)

def load_data(partition_id: int, dataset_dir: str) -> tuple[np.ndarray, ...]:
    dataset = get_node_dataset(dataset_dir, partition_id)
    return dataset

def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    experiments_dir = config.get("experiments-dir")
    network_name = config.get("network-name")
    network_scenario = config.get("network-scenario")
    simulation_run = config.get("simulation-run")
    run_dir = f"{network_name}/{network_scenario}/run_{simulation_run}"
    # Save path is based on the current directory
    save_path = Path.cwd() / f"{experiments_dir}/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=True)

    return save_path, run_dir

def save_run_config(config: UserConfig, save_path: Path) -> None:
    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)