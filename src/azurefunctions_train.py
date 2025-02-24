from utils.model_creators import create_LSTM
from gossiplearning.config import Config

from azurefunctions_utils import load_dataset

from keras.api.callbacks import ModelCheckpoint
from tensorflow.python.keras import Model
from matplotlib import colors as mcolors
from keras.api.models import load_model
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
    description="Train models for Azure function traces"
  )
  parser.add_argument(
    "-f", "--base_folder", 
    help="Paths to the base folder", 
    type=str,
    required=True
  )
  parser.add_argument(
    "-m", "--modes", 
    help="Training mode(s)", 
    type = str,
    nargs = "+",
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
  parser.add_argument(
    "--plot_single_history", 
    help="True to plot the history of each model", 
    default=False,
    action="store_true"
  )
  args, _ = parser.parse_known_args()
  return args


def already_trained(
    output_folder: str, model_keyword: str
  ) -> Tuple[bool, str, str]:
  history_file = os.path.join(output_folder, f"{model_keyword}_history.csv")
  model_file = os.path.join(output_folder, f"{model_keyword}.keras")
  at = os.path.exists(history_file) and os.path.exists(model_file)
  return at, history_file, model_file


def compute_and_plot_predictions(
    X_Y_data: dict, 
    model,
    output_folder: str,
    node: str
  ) -> dict:
  # compute predictions
  predictions = {}
  for data_key, (X_data, _) in X_Y_data.items():
    predictions[data_key] = model.predict(X_data)
  # plot
  nrows = len(X_Y_data)
  _, axs = plt.subplots(
    nrows = nrows, 
    ncols = 1,
    figsize = (7 * nrows, 8)
  )
  idx = 0
  for data_key, Y_pred in predictions.items():
    ax = axs if nrows == 1 else axs[idx]
    Y_data = X_Y_data[data_key][1]
    ax.plot(
      range(len(Y_data)),
      Y_data,
      color = mcolors.TABLEAU_COLORS["tab:blue"],
      marker = "."
    )
    ax.plot(
      range(len(Y_data)), 
      Y_pred,
      color = mcolors.TABLEAU_COLORS["tab:orange"],
      marker = ".",
      alpha = 0.5
    )
    ax.set_ylabel(data_key)
    idx += 1
  # save figure
  if output_folder is not None:
    plt.savefig(
      os.path.join(output_folder, f"predictions_{node}.png"),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()
  return predictions


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


def plot_multinode_history(
    histories: pd.DataFrame, output_folder: str, keyword: str
  ):
  colors = list(mcolors.TABLEAU_COLORS.values())
  centralized_exists = "centralized" in histories["node"].unique()
  _, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (20,8))
  for node, history_df in histories.groupby("node"):
    color = "k" if node == "centralized" else (
      colors[0] if centralized_exists else colors[int(node)]
    )
    linewidth = 2 if node == "centralized" else 1
    linestyle = "solid" if node == "centralized" else "dashed"
    label = node
    history_df.plot(
      x = "iter",
      y = "mape", 
      ax = axs[0,0],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    history_df.plot(
      x = "iter",
      y = "val_mape", 
      ax = axs[0,1],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    history_df.plot(
      x = "iter",
      y = "mse", 
      ax = axs[1,0],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
    history_df.plot(
      x = "iter",
      y = "val_mse", 
      ax = axs[1,1],
      label = label,
      grid = True,
      color = color,
      linewidth = linewidth,
      linestyle = linestyle
    )
  # plot average over local models
  local = histories[histories["node"]!="centralized"].groupby("iter").mean(
    numeric_only = True
  )
  color = colors[0] if "centralized" in histories["node"].unique() else "r"
  linewidth = 2
  linestyle = "solid"
  local.plot(
    y = "mape", 
    ax = axs[0,0],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  local.plot(
    y = "val_mape", 
    ax = axs[0,1],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  local.plot(
    y = "mse", 
    ax = axs[1,0],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  local.plot(
    y = "val_mse", 
    ax = axs[1,1],
    label = "local average",
    grid = True,
    color = color,
    linewidth = linewidth,
    linestyle = linestyle
  )
  # axis properties
  axs[0,0].set_ylabel("MAPE", fontsize = 14)
  axs[1,0].set_ylabel("MSE", fontsize = 14)
  axs[0,0].set_title("Training History", fontsize = 14)
  axs[0,1].set_title("Validation History", fontsize = 14)
  axs[0,1].legend(
    fontsize = 14, loc = "center left", bbox_to_anchor = (1, 0)
  )
  plt.savefig(
    os.path.join(output_folder, f"{keyword}_history.png"),
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
  train_data, validation_data, _ = dataset
  X_train = train_data[0]
  Y_train = train_data[1]
  X_val = validation_data[0]
  Y_val = validation_data[1]
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
    nodes_dataset: dict, 
    output_folder: str,
    plot_single_history: bool
  ) -> Tuple[dict, pd.DataFrame]:
  # loop over nodes
  models = {}
  histories = pd.DataFrame()
  for node in nodes_dataset:
    model, history = train_one_model(
      config = config, 
      model_creator = model_creator, 
      dataset = nodes_dataset[node], 
      checkpoint_path = os.path.join(output_folder, f"{node}_single.keras")
    )
    # save model
    models[node] = model
    # plot and save history
    history_df = pd.DataFrame(history)
    history_df["iter"] = history_df.index
    history_df["node"] = [node] * len(history_df)
    history_df.to_csv(
      os.path.join(output_folder, f"{node}_single_history.csv"), index = False
    )
    if plot_single_history:
      plot_history(history_df, output_folder, f"{node}_single")
    histories = pd.concat([histories, history_df], ignore_index = True)
    # compute, plot and save predictions
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = nodes_dataset[
      node
    ]
    predictions = compute_and_plot_predictions(
      {
        "train": (X_train, Y_train), 
        "val": (X_val, Y_val), 
        "test": (X_test, Y_test)
      }, 
      models[node], 
      output_folder, 
      f"{node}_single"
    )
    predictions_json = {
      "train": predictions["train"].tolist(),
      "val": predictions["val"].tolist(),
      "test": predictions["test"].tolist()
    }
    with open(
        os.path.join(output_folder, f"{node}_single_pred.json"), "w"
      ) as ost:
      ost.write(json.dumps(predictions_json, indent = 2))
  return models, histories


def train_centralized_model(
    config: Config, 
    model_creator: functools.partial, 
    centralized_dataset: list, 
    output_folder: str,
    plot_single_history: bool
  ) -> Tuple[dict, pd.DataFrame]:
  # train
  model, history = train_one_model(
    config = config,
    model_creator = model_creator,
    dataset = centralized_dataset,
    checkpoint_path = os.path.join(output_folder, "centralized.keras")
  )
  # plot and save history
  history_df = pd.DataFrame(history)
  history_df["node"] = ["centralized"] * len(history_df)
  history_df["iter"] = history_df.index
  history_df.to_csv(
    os.path.join(output_folder, "centralized_history.csv"), index = False
  )
  if plot_single_history:
    plot_history(history_df, output_folder, "centralized")
  # compute, plot and save predictions
  predictions = compute_and_plot_predictions(
    {
      "train": centralized_dataset[0],
      "val": centralized_dataset[1],
      "test": centralized_dataset[2]
    },
    model,
    output_folder, "centralized"
  )
  predictions_json = {
      "train": predictions["train"].tolist(),
      "val": predictions["val"].tolist(),
      "test": predictions["test"].tolist()
    }
  with open(os.path.join(output_folder, "centralized_pred.json"), "w") as ost:
    ost.write(json.dumps(predictions_json, indent = 2))
  # return
  return model, history_df


def run_single_training_experiment(
    base_folder: str,
    simulation: int,
    config: Config, 
    model_creator: functools.partial, 
    mode: str = "centralized",
    plot_single_history: bool = False
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
    is_already_trained, history_file, model_file = already_trained(
      output_folder, "centralized"
    )
    history_df = None
    if not is_already_trained:
      dataset = load_dataset(data_folder, "centralized")
      models["centralized"], history_df = train_centralized_model(
        config = config,
        model_creator = model_creator,
        centralized_dataset = dataset,
        output_folder = output_folder,
        plot_single_history = plot_single_history
      )
    else:
      print(f"  {mode} results for simulation {simulation} already exist!")
      models["centralized"] = load_model(model_file)
      history_df = pd.read_csv(history_file)
    histories = pd.concat([histories, history_df], ignore_index = True)
  elif mode == "local":
    # local training for all nodes
    dataset = {}
    for node in range(config.n_nodes):
      is_already_trained, history_file, model_file = already_trained(
        output_folder, f"{node}_single"
      )
      if not is_already_trained:
        dataset[node] = load_dataset(data_folder, node)
      else:
        print(
          f"  {mode}-{node} results for simulation {simulation} already exist!"
        )
        models[node] = None#load_model(model_file)
        history_df = pd.read_csv(history_file)
        histories = pd.concat([histories, history_df], ignore_index = True)
    if len(dataset) > 0:
      l_models, l_histories = train_local_models(
        config = config,
        model_creator = model_creator,
        nodes_dataset = dataset,
        output_folder = output_folder,
        plot_single_history = plot_single_history
      )
      models = {**models, **l_models}
      histories = pd.concat([histories, l_histories], ignore_index = True)
    # plot history of different nodes
    plot_multinode_history(histories, output_folder, "local")
  return models, histories


def train(
    config_file: str, 
    base_folder: str,
    modes: str,
    seed: int,
    simulations: list,
    plot_single_history: bool
  ) -> Tuple[dict, pd.DataFrame]:
  # load configuration and define model creator
  config, model_creator = prepare_training(config_file)
  # build base i/o folder
  n = config.n_nodes
  k = config.connectivity
  t = config.data_preparation.time_window
  io_folder = os.path.join(
    base_folder,
    f"azurefunctions-dataset2019/{n}n_{k}k_{t}min/seed{seed}"
  )
  # loop over simulations
  models = {}
  histories = pd.DataFrame()
  for simulation in simulations:
    sim_models = {}
    sim_histories = pd.DataFrame()
    for mode in modes:
      mode_models, mode_histories = run_single_training_experiment(
        base_folder = io_folder,
        simulation = simulation,
        config = config,
        model_creator = model_creator,
        mode = mode,
        plot_single_history = plot_single_history
      )
      mode_histories["mode"] = [mode] * len(mode_histories)
      # save
      sim_models = {**sim_models, **mode_models}
      sim_histories = pd.concat(
        [sim_histories, mode_histories], 
        ignore_index = True
      )
    # plot simulation history
    plot_multinode_history(sim_histories, io_folder, f"{simulation}_all")
  return models, histories


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  config_file = args.config_file
  base_folder = args.base_folder
  modes = args.modes
  seed = args.seed
  simulations = args.simulations
  plot_single_history = args.plot_single_history
  # config_file = "azurefunctions_config.json"
  # base_folder = "../experiments"
  # modes = ["local", "centralized"]
  # seed = 4850
  # simulations = 0
  if not isinstance(modes, list):
    modes = [modes]
  if not isinstance(simulations, list) and simulations != "all":
    simulations = [simulations]
  # run
  _ = train(
    config_file, base_folder, modes, seed, simulations, plot_single_history
  )
