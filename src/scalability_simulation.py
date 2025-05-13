import functools
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

from gossiplearning.config import Config
from gossiplearning.models import MergeStrategy, StopCriterion
from gossiplearning.weight import weight_by_dataset_size
from utils.data import get_common_test_set
from utils.gossip_training import (
    get_node_dataset,
    round_trip_fn,
    run_simulation,
)
from utils.model_creators import create_LSTM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

with open("scalability_simulation_config.json", "r") as f:
    config_json = json.load(f)
    config = Config.model_validate(config_json)

model_creator = functools.partial(create_LSTM, config=config)

prefix = "porto"
ds_name = ""

SIMULATIONS_PER_CONFIG = 10


def model_transmission_fn(i: int, j: int) -> int:
    return random.randint(25, 35)

worker_jobs = []

for nodes in [50, 100]:
    n = nodes
    k = 3
    datasets_folder = Path(f"data/datasets/porto_{n}n_{k}k")
    networks_folder = Path(f"data/networks/porto_{n}n_{k}k")

    config_json["n_nodes"] = n
    config_json["workspace_dir"] = f"experiments/scalability/{n}n_{k}k"
    config = Config.model_validate(config_json)

    for i in range(SIMULATIONS_PER_CONFIG):
        all_history_plot = (
            Path(config.workspace_dir) / str(i) / "plots" / "history" / "all.jpg"
        )
        if all_history_plot.exists():
            continue
        worker_jobs.append((config, i, datasets_folder, networks_folder))


def run_worker(data: tuple[Config, int, Path, Path]) -> None:
    config, i, ds, nw = data

    node_data_fn = functools.partial(
        get_node_dataset,
        base_folder=ds,
        simulation_number=i,
        ds_name=ds_name,
    )

    get_test_set = functools.partial(
        get_common_test_set,
        node_data_fn=node_data_fn,
        n_nodes=config.n_nodes,
        perc=0.1,
    )

    run_simulation(
        config=config,
        simulation_number=i,
        network_folder=nw / str(i),
        round_trip_fn=round_trip_fn,
        model_transmission_fn=model_transmission_fn,
        node_data_fn=node_data_fn,
        model_creator=model_creator,
        get_test_set=get_test_set,
        weight_fn=weight_by_dataset_size,
    )


if __name__ == "__main__":
    with mp.Pool(int(mp.cpu_count()/2)) as p:
        p.map(run_worker, worker_jobs)

