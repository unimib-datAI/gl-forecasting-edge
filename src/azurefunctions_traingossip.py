from utils.gossip_training import round_trip_fn, run_simulation
from gossiplearning.weight import weight_by_dataset_size
from utils.data import get_common_test_set
from gossiplearning.config import Config

from azurefunctions_train import prepare_training
from azurefunctions_utils import load_dataset

from pathlib import Path
import functools
import random
import os


def get_node_dataset(
    node: int, base_folder: str, simulation_number: int
  ) -> dict:
  # build complete foldername
  foldername = os.path.join(base_folder, str(simulation_number))
  # load data
  (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = load_dataset(
    foldername, node
  )
  # arrange in dictionary
  dataset = {
    "X_train": X_train, 
    "Y_train": Y_train, 
    "X_val": X_val, 
    "Y_val": Y_val, 
    "X_test": X_test, 
    "Y_test": Y_test
  }
  return dataset


def model_transmission_fn(i: int = 25, j: int = 35) -> int:
  return random.randint(25, 35)


def run_worker(
    data: tuple[Config, int, str, str], model_creator: functools.partial
  ):
  config, i, datasets_folder, networks_folder = data
  # create folder to plot training history
  all_history_plot = os.path.join(
    config.workspace_dir, str(i), "plots/history/all.jpg"
  )
  if os.path.exists(all_history_plot):
    print(f"WARNING: plot {all_history_plot} already exists!")
    return
  # define function to load dataset
  node_data_fn = functools.partial(
    get_node_dataset,
    base_folder = datasets_folder,
    simulation_number = i
  )
  # define function to create a common test set
  get_test_set = functools.partial(
    get_common_test_set,
    node_data_fn = node_data_fn,
    n_nodes = config.n_nodes,
    perc = 0.1,
  )
  # run simulation
  run_simulation(
    config = config,
    simulation_number = i,
    network_folder = Path(os.path.join(networks_folder, str(i))),
    round_trip_fn = round_trip_fn,
    model_transmission_fn = model_transmission_fn,
    node_data_fn = node_data_fn,
    model_creator = model_creator,
    get_test_set = get_test_set,
    weight_fn = weight_by_dataset_size,
  )


if __name__ == "__main__":
  config, model_creator = prepare_training("azurefunctions_config.json")
  i = 0
  datasets_folder = "../experiments/azurefunctions-dataset2019/10n_k3_15min/seed4850"
  networks_folder = "../data/networks/porto_10n_3k/seed4850"
  # run
  run_worker((config, i, datasets_folder, networks_folder), model_creator)
