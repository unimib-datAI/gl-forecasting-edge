from utils.model_creators import create_LSTM
from gossiplearning.config import Config

from azurefunctions_utils import load_dataset, save_dataset

from keras.api.callbacks import ModelCheckpoint
from tensorflow.python.keras import Model
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import numpy as np
import functools
import argparse
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(
    description="Prepare dataset(s) to train models for Azure function traces"
  )
  parser.add_argument(
    "-f", "--base_folder", 
    help="Paths to the base folder", 
    type=str,
    required=True
  )
  parser.add_argument(
    "-m", "--mode", 
    help="Training mode", 
    type = str,
    choices = ["centralized", "local"],
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


def compute_and_plot_predictions(
    X_train: np.ndarray, 
    Y_train: np.ndarray, 
    X_val: np.ndarray, 
    Y_val: np.ndarray, 
    X_test: np.ndarray, 
    Y_test: np.ndarray, 
    model,
    output_folder: str,
    node: str
  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  Y_train_pred = model.predict(X_train)
  Y_val_pred = model.predict(X_val)
  Y_test_pred = model.predict(X_test)
  _, axs = plt.subplots(
    nrows = 3, 
    ncols = 1,
    figsize = (20,8)
  )
  # train
  axs[0].plot(
    range(len(Y_train)), 
    Y_train,
    color = mcolors.TABLEAU_COLORS["tab:blue"],
    marker = "."
  )
  axs[0].plot(
    range(len(Y_train)), 
    Y_train_pred,
    color = mcolors.TABLEAU_COLORS["tab:orange"],
    marker = ".",
    alpha = 0.5
  )
  # validation
  axs[1].plot(
    range(len(Y_val)), 
    Y_val,
    color = mcolors.TABLEAU_COLORS["tab:blue"],
    marker = "."
  )
  axs[1].plot(
    range(len(Y_val)), 
    Y_val_pred,
    color = mcolors.TABLEAU_COLORS["tab:orange"],
    marker = ".",
    alpha = 0.5
  )
  # test
  axs[2].plot(
    range(len(Y_test)), 
    Y_test,
    color = mcolors.TABLEAU_COLORS["tab:blue"],
    marker = "."
  )
  axs[2].plot(
    range(len(Y_test)), 
    Y_test_pred,
    color = mcolors.TABLEAU_COLORS["tab:orange"],
    marker = ".",
    alpha = 0.5
  )
  plt.savefig(
    os.path.join(output_folder, f"predictions_{node}.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  return Y_train_pred, Y_val_pred, Y_test_pred


def plot_history(history_df: pd.DataFrame, output_folder: str, node: str):
  # plot MAPE history
  history_df[["mape", "val_mape"]].plot()
  plt.savefig(
    os.path.join(output_folder, f"mape_{node}.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()
  # plot MSE history
  history_df[["mse", "val_mse"]].plot()
  plt.savefig(
    os.path.join(output_folder, f"mse_{node}.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


def prepare_training(config_file: str) -> Tuple[Config, functools.partial]:
  # load and validate configuration
  config = None
  with open(config_file, "r") as f:
    config = Config.model_validate(json.load(f))
  # define model creator
  model_creator = functools.partial(
      create_LSTM,
      config = config,
  )
  print(model_creator().summary())
  return config, model_creator


def train_one_model(
    config: Config, 
    model_creator: functools.partial, 
    dataset: list, 
    checkpoint_path: str
  ) -> Tuple[Model, dict]:
  # prepare model and output files
  model = model_creator()
  model_checkpoint = ModelCheckpoint(
    filepath = checkpoint_path,
    save_best_only = True,
    monitor = "val_loss",
    mode = "min",
  )
  # extract data
  (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = dataset
  # train
  history = model.fit(
    X_train,
    [Y_train[:, fn] for fn in range(config.training.n_output_vars)],
    validation_data = (
      X_val,
      [Y_val[:, fn] for fn in range(config.training.n_output_vars)],
    ),
    validation_batch_size = config.training.batch_size,
    verbose = 1,
    callbacks = [
      # early_stopping,
      model_checkpoint,
    ],
    epochs = 100,
    batch_size = config.training.batch_size,
    shuffle = config.training.shuffle_batch,
    # use_multiprocessing = False,
  ).history
  return model, history


def train_local_models(
    config: Config, 
    model_creator: functools.partial, 
    prepared_nodes_dataset: dict, 
    output_folder: str
  ) -> Tuple[dict, pd.DataFrame]:
  # loop over nodes
  models = {}
  histories = pd.DataFrame()
  for node in prepared_nodes_dataset:
    model, history = train_one_model(
      config = config, 
      model_creator = model_creator, 
      dataset = prepared_nodes_dataset[node], 
      checkpoint_path = os.path.join(output_folder, f"{node}_single.h5")
    )
    # save model
    models[node] = model
    # plot and save history
    history_df = pd.DataFrame(history)
    plot_history(history_df, output_folder, node)
    history_df.to_csv(os.path.join(output_folder, f"{node}_history.csv"))
    history_df["node"] = [node] * len(history_df)
    histories = pd.concat([histories, history_df], ignore_index = True)
    # compute, plot and save predictions
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = prepared_nodes_dataset[
      node
    ]
    Y_train_pred, Y_val_pred, Y_test_pred = compute_and_plot_predictions(
      X_train, Y_train, 
      X_val, Y_val, 
      X_test, Y_test, 
      models[node], 
      output_folder, 
      node
    )
    predictions_json = {
      "train": Y_train_pred.tolist(),
      "val": Y_val_pred.tolist(),
      "test": Y_test_pred.tolist()
    }
    with open(os.path.join(output_folder, f"{node}_pred.json"), "w") as ost:
      ost.write(json.dumps(predictions_json, indent = 2))
  return models, histories


def train_centralized_model(
    config: Config, 
    model_creator: functools.partial, 
    centralized_dataset: list, 
    output_folder: str
  ) -> Tuple[dict, pd.DataFrame]:
  # load data
  train_data, val_data, test_data = centralized_dataset
  # train
  model, history = train_one_model(
    config = config,
    model_creator = model_creator,
    dataset = [*train_data, *val_data, *test_data],
    checkpoint_path = os.path.join(output_folder, "centralized.h5")
  )
  # plot and save history
  history_df = pd.DataFrame(history)
  plot_history(history_df, output_folder, "centralized")
  history_df.to_csv(os.path.join(output_folder, f"centralized_history.csv"))
  # compute, plot and save predictions
  Y_train_pred, Y_val_pred, Y_test_pred = compute_and_plot_predictions(
    train_data[0], train_data[1],
    val_data[0], val_data[1],
    test_data[0], test_data[1],
    model,
    output_folder, "centralized"
  )
  predictions_json = {
    "train": Y_train_pred.tolist(),
    "val": Y_val_pred.tolist(),
    "test": Y_test_pred.tolist()
  }
  with open(os.path.join(output_folder, f"centralized_pred.json"), "w") as ost:
    ost.write(json.dumps(predictions_json, indent = 2))
  # return
  return model, history_df


def run_single_training_experiment(
    base_folder: str,
    simulation: int,
    config: Config, 
    model_creator: functools.partial, 
    mode: str = "centralized"
  ) -> Tuple[dict, pd.DataFrame]:
  # build name of data folder
  data_folder = os.path.join(base_folder, str(simulation))
  # build output folder
  output_folder = os.path.join(data_folder, "results")
  os.makedirs(output_folder, exist_ok = True)
  # load data and train according to mode
  models = {}
  histories = pd.DataFrame()
  if mode == "centralized":
    # centralized training
    dataset = load_dataset(data_folder, "centralized")
    models["centralized"], history_df = train_centralized_model(
      config = config,
      model_creator = model_creator,
      centralized_dataset = dataset,
      output_folder = output_folder
    )
    history_df["node"] = "centralized"
    histories = pd.concat([histories, history_df], ignore_index = True)
  elif mode == "local":
    # local training for all nodes
    dataset = {
      node: load_dataset(data_folder, node) for node in range(config.n_nodes)
    }
    models, histories = train_local_models(
      config = config,
      model_creator = model_creator,
      prepared_nodes_dataset = dataset,
      output_folder = output_folder
    )
    # plot history of different nodes
    _, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (20,8))
    for node, history_df in histories.groupby("node"):
      to_plot = history_df.reset_index()
      to_plot["mape"].plot(
        ax = axs[0,0],
        label = node,
        grid = True
      )
      to_plot["val_mape"].plot(
        ax = axs[0,1],
        label = node,
        legend = False,
        grid = True
      )
      to_plot["mse"].plot(
        ax = axs[1,0],
        label = node,
        legend = False,
        grid = True
      )
      to_plot["val_mse"].plot(
        ax = axs[1,1],
        label = node,
        legend = False,
        grid = True
      )
    axs[0,0].set_ylabel("MAPE", fontsize = 14)
    axs[1,0].set_ylabel("MSE", fontsize = 14)
    axs[0,0].set_title("Training History", fontsize = 14)
    axs[0,1].set_title("Validation History", fontsize = 14)
    plt.savefig(
      os.path.join(output_folder, "history_local.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  return models, histories


def train(
    config_file: str, 
    base_folder: str,
    mode: str,
    seed: int,
    simulations: list
  ) -> list:
  # load configuration and define model creator
  config, model_creator = prepare_training(config_file)
  # build base i/o folder
  n = config.n_nodes
  k = config.connectivity
  t = config.data_preparation.time_window
  io_folder = os.path.join(
    base_folder,
    f"azurefunctions-dataset2019/{n}n_k{k}_{t}min/seed{seed}"
  )
  # loop over simulations
  output_folders = []
  for simulation in simulations:
    output_folder = run_single_training_experiment(
      base_folder = io_folder,
      simulation = simulation,
      config = config,
      model_creator = model_creator,
      mode = mode
    )
    output_folders.append(output_folder)
  return output_folders


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  config_file = args.config_file
  base_folder = args.base_folder
  mode = args.mode
  seed = args.seed
  simulations = args.simulations
  if not isinstance(simulations, list) and simulations != "all":
    simulations = [simulations]
  # run
  _ = train(config_file, base_folder, mode, seed, simulations)
