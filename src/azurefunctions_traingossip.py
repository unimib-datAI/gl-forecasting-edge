from utils.gossip_training import round_trip_fn, run_simulation
from gossiplearning.weight import weight_by_dataset_size
from utils.data import get_common_test_set
from gossiplearning.config import Config

from azurefunctions_train import prepare_training
from azurefunctions_utils import load_dataset

from datetime import datetime
from pathlib import Path
import tensorflow as tf
import numpy as np
import functools
import argparse
import random
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(
    description="Train models for Azure function traces with Gossip Learning"
  )
  parser.add_argument(
    "-f", "--base_folder", 
    help="Paths to the base folder", 
    type=str,
    required=True
  )
  parser.add_argument(
    "-c", "--config_file", 
    help="Config file", 
    type=str,
    default="azurefunctions_config.json"
  )
  parser.add_argument(
    "--seed", 
    help="Seed for random number generation", 
    type=int,
    default=4850
  )
  parser.add_argument(
    "--simulations", 
    help="Simulation indices", 
    type=int,
    nargs="+",
    default=0
  )
  args, _ = parser.parse_known_args()
  return args


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
  args = parse_arguments()
  base_folder = args.base_folder
  config_file = args.config_file
  simulations = args.simulations
  seed = args.seed
  if not isinstance(simulations, list) and simulations != "all":
    simulations = [simulations]
  # load and validate configuration; define model creator
  config, model_creator = prepare_training("azurefunctions_config.json")
  # loop over simulations
  for i in simulations:
    print(80 * "-")
    print(f"Simulation {i} starts on {datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')}")
    print(80 * "-")
    n = config.n_nodes
    k = config.connectivity
    t = config.data_preparation.time_window
    # datasets and networks folders
    datasets_folder = os.path.join(
      base_folder,
      f"experiments/azurefunctions-dataset2019/{n}n_{k}k_{t}min/seed{seed}"
    )
    networks_folder = os.path.join(
      base_folder, 
      f"data/networks/porto_{n}n_{k}k/seed{seed}"
    )
    # gossip folder
    config.workspace_dir = os.path.join(datasets_folder, "gossip")
    # set seed
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    # run
    run_worker((config, i, datasets_folder, networks_folder), model_creator)
