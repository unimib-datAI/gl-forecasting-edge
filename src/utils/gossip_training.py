import dataclasses
import json
import random
from pathlib import Path
from typing import Callable

import networkx as nx
import numpy as np
from keras import Model

from gossiplearning import Simulator, plot_history
from gossiplearning.config import Config
from gossiplearning.links_strategy import (
    add_node_and_links_to_config,
)
from gossiplearning.models import (
    TimeFn,
    NodeDataFn,
    Dataset,
    LabelledData,
    NodeWeightFn,
)
from gossiplearning.utils import NpEncoder


def get_node_dataset(
    node_index: int,
    base_folder: Path,
    simulation_number: int,
    ds_name: str,
) -> Dataset:
    dataset: Dataset = np.load(
        base_folder / str(simulation_number) / ds_name / f"node_{node_index}.npz"
    )

    return {
        "X_train": dataset["X_train"],
        "Y_train": dataset["Y_train"],
        "X_val": dataset["X_val"],
        "Y_val": dataset["Y_val"],
        "X_test": dataset["X_test"],
        "Y_test": dataset["Y_test"],
    }


def get_static_node_dataset(node_index: int, base_folder: Path) -> str:
    return str(base_folder / f"node_{node_index}.npz")


def round_trip_fn(i: int, j: int) -> int:
    return 1


def model_transmission_fn(i: int, j: int) -> int:
    return random.randint(25, 35)


def run_simulation(
    config: Config,
    simulation_number: int,
    network_folder: Path,
    round_trip_fn: TimeFn,
    model_transmission_fn: TimeFn,
    node_data_fn: NodeDataFn,
    model_creator: Callable[[], Model],
    get_test_set: Callable[[], LabelledData],
    weight_fn: NodeWeightFn,
) -> None:
    network = nx.read_adjlist(network_folder / "adj_list.txt", nodetype=int)
    extended_config = add_node_and_links_to_config(
        config, network, round_trip_fn, model_transmission_fn
    )
    extended_config.workspace_dir = Path(config.workspace_dir) / str(simulation_number)

    workspace_path = Path(extended_config.workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)

    with (workspace_path / "config.json").open("w") as f:
        json.dump(extended_config.model_dump(), f, indent=True)

    test_set = get_test_set()
    with (workspace_path / "common_test_set.json").open("w") as f:
        test_set_json = {
            "X_test": test_set[0].tolist(),
            "Y_test": test_set[1].tolist()
        }
        f.write(json.dumps(test_set_json, indent = 2))

    simulator = Simulator(
        create_model=model_creator,
        config=extended_config,
        node_data_fn=node_data_fn,
        test_set=test_set,
        weight_fn=weight_fn,
    )

    models_folder = Path(extended_config.workspace_dir) / "models"
    models_folder.mkdir(parents=True, exist_ok=True)
    
    history = simulator.run_training_simulation()

    with (workspace_path / f"history.json").open("w") as file:
        json.dump(dataclasses.asdict(history), file, indent=True, cls=NpEncoder)

    history_plots_folder = Path(extended_config.workspace_dir) / "plots" / "history"
    history_plots_folder.mkdir(parents=True, exist_ok=True)

    for i in range(config.n_nodes):
        plot_file = history_plots_folder / f"node_{i}.jpg"
        plot_history(history, file=str(plot_file), only_nodes=tuple([i]))

    plot_file = history_plots_folder / "all.jpg"
    plot_history(history, file=str(plot_file), only_nodes=None)
