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

with open("config.json", "r") as f:
    config_json = json.load(f)
    config = Config.model_validate(config_json)

model_creator = functools.partial(create_LSTM, config=config)

prefix = "porto"
ds_name = "4in_notscaled"

SIMULATIONS_PER_CONFIG = 10


def model_transmission_fn(i: int, j: int) -> int:
    return random.randint(25, 35)


# strategy, epochs per update, target probability, perc sent weights
combinations = (
    (MergeStrategy.AGE_WEIGHTED, 5, 0.1),
    (MergeStrategy.SIMPLE_AVG, 5, 0.1),
    (MergeStrategy.OVERWRITE, 5, 0.1),
    (MergeStrategy.IMPROVED_OVERWRITE, 5, 0.1),
    (MergeStrategy.AGE_WEIGHTED, 5, 1),
    (MergeStrategy.SIMPLE_AVG, 5, 1),
    (MergeStrategy.OVERWRITE, 5, 1),
    (MergeStrategy.IMPROVED_OVERWRITE, 5, 1),
)
exp = 177

worker_jobs = []

for ms, patience, min_delta in combinations:
    n = 10
    k = 3
    datasets_folder = Path(f"data/datasets/porto_{n}n_{k}k")
    networks_folder = Path(f"data/networks/porto_{n}n_{k}k")

    config_json["n_nodes"] = n
    config_json["workspace_dir"] = f"experiments/{prefix}_{exp}"
    config_json["training"]["merge_strategy"] = MergeStrategy.AGE_WEIGHTED
    config_json["training"]["epochs_per_update"] = 4
    config_json["training"]["finetuning_epochs"] = 0
    config_json["training"]["target_probability"] = 1

    config_json["training"]["num_merged_models"] = 1
    config_json["training"]["perc_sent_weights"] = 1
    config_json["training"]["serialize_optimizer"] = False

    # config_json["training"]["stop_criterion"] = StopCriterion.FIXED_UPDATES.value
    # config_json["training"]["fixed_updates"] = 25

    config_json["training"]["stop_criterion"] = StopCriterion.NO_IMPROVEMENTS.value
    config_json["training"]["patience"] = patience
    config_json["training"]["min_delta"] = min_delta

    config = Config.model_validate(config_json)

    for i in range(SIMULATIONS_PER_CONFIG):
        worker_jobs.append((config, i, datasets_folder, networks_folder))

    exp += 1


def run_worker(data: tuple[Config, int, Path, Path]) -> None:
    config, i, ds, nw = data

    all_history_plot = (
        Path(config.workspace_dir) / str(i) / "plots" / "history" / "all.jpg"
    )
    if all_history_plot.exists():
        return

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
    for job in worker_jobs:
        p = mp.Process(
            target=run_worker,
            args=(job,),
        )

        p.start()

        p.join()
