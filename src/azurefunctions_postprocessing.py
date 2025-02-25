from azurefunctions_train import compute_and_plot_predictions
from azurefunctions_utils import load_dataset

from utils.metrics import compute_metrics

from matplotlib import colors as mcolors
from keras.api.models import load_model
import matplotlib.pyplot as plt
from typing import Tuple
import pandas as pd
import numpy as np
import json
import os


def compute_predictions(
    base_folder: str, simulation_index: int
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  models_folder = os.path.join(
    base_folder, "gossip", str(simulation_index), "models"
  )
  data_folder = os.path.join(
    base_folder, str(simulation_index)
  )
  common_test_file = os.path.join(
    base_folder, "gossip", str(simulation_index), "common_test_set.json"
  )
  plot_folder = os.path.join(
    base_folder, "gossip", str(simulation_index), "plots"
  )
  # load models and nodes-specific data
  models = {}
  all_X_Y_data = {}
  for model_filename in os.listdir(models_folder):
    # get node id
    node = int(model_filename.split(".")[0])
    # load model
    model = load_model(os.path.join(models_folder, model_filename))
    models[node] = model
    # load data
    train, val, test = load_dataset(data_folder, node)
    all_X_Y_data[node] = {
      "train": train,
      "val": val,
      "test": test
    }
  # load common test set
  X_test, Y_test = None, None
  with open(common_test_file, "r") as istream:
    common_test_set = json.load(istream)
    X_test = np.array(common_test_set["X_test"])
    Y_test = np.array(common_test_set["Y_test"])
  # compute and plot predictions
  all_predictions = pd.DataFrame()
  all_metrics = {"key": [], "node": [], "metrics": []}
  for node, model in models.items():
    X_Y_data = all_X_Y_data[node]
    X_Y_data["common_test"] = [X_test, Y_test]
    predictions = compute_and_plot_predictions(
      X_Y_data, models[node], plot_folder, str(node)
    )
    # get train, val, test
    for key, pred in predictions.items():
      Y_real = X_Y_data[key][1]
      # save
      df = pd.DataFrame({
        "real": [float(y[0]) for y in Y_real],
        "pred": [float(y[0]) for y in pred],
        "key": [key] * len(Y_real),
        "node": [node] * len(Y_real),
      })
      all_predictions = pd.concat([all_predictions, df], ignore_index = True)
      # compute metrics
      metrics = compute_metrics(Y_real, pred)
      all_metrics["key"].append(key)
      all_metrics["node"].append(node)
      all_metrics["metrics"].append(metrics)
  # extract some surely-relevant metrics
  all_metrics = pd.DataFrame(all_metrics)
  all_metrics["mape"] = [m.mape * 100 for m in all_metrics["metrics"]]
  all_metrics["mse"] = [m.mse for m in all_metrics["metrics"]]
  return all_predictions, all_metrics


def load_existing_predictions(
    data_folder: str, nodes: list
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # loop over nodes
  all_predictions = pd.DataFrame()
  all_metrics = {"key": [], "node": [], "metrics": []}
  for node in nodes:
    # load data
    data = load_dataset(data_folder, node)
    # load predictions
    pred_file = ""
    if node != "centralized":
      pred_file = os.path.join(data_folder, f"results/{node}_single_pred.json")
    else:
      pred_file = os.path.join(data_folder, "results/centralized_pred.json")
    predictions = {}
    with open(pred_file, "r") as istream:
      predictions = json.load(istream)
    # get train, val, test
    for idx, (key, pred) in enumerate(predictions.items()):
      Y_real = data[idx][1]
      # save
      df = pd.DataFrame({
        "real": [float(y[0]) for y in Y_real],
        "pred": [float(y[0]) for y in pred],
        "key": [key] * len(Y_real),
        "node": [node] * len(Y_real),
      })
      all_predictions = pd.concat([all_predictions, df], ignore_index = True)
      # compute metrics
      metrics = compute_metrics(Y_real, pred)
      all_metrics["key"].append(key)
      all_metrics["node"].append(node)
      all_metrics["metrics"].append(metrics)
  # extract some surely-relevant metrics
  all_metrics = pd.DataFrame(all_metrics)
  all_metrics["mape"] = [m.mape * 100 for m in all_metrics["metrics"]]
  all_metrics["mse"] = [m.mse for m in all_metrics["metrics"]]
  return all_predictions, all_metrics


def plot_metrics(
    all_metrics: pd.DataFrame, 
    experiment_idx: int, 
    relevant_metrics: list,
    output_folder: str = None
  ):
  # rename 'centralized' to 'C' (if available)
  if "centralized" in all_metrics["node"]:
    all_metrics.replace("centralized", "C", inplace = True)
  # get maximum and minimum number of nodes
  minnode = min([n for n in all_metrics["node"].unique() if n != "C"])
  maxnode = max([n for n in all_metrics["node"].unique() if n != "C"])
  # plot
  nrows = len(relevant_metrics)
  ncols = len(all_metrics["key"].unique())
  _, axs = plt.subplots(
    nrows = nrows,
    ncols = ncols,
    figsize = (8 * ncols, 3 * nrows),
    sharex = True,
    sharey = "row"
  )
  cidx = 0
  for key, metrics in all_metrics.groupby("key"):
    cax = axs if ncols == 1 else axs[:,cidx]
    ridx = 0
    for metric_name in relevant_metrics:
      rax = cax if nrows == 1 else cax[ridx]
      metrics.plot.bar(
        x = "node",
        y = metric_name,
        rot = 0,
        fontsize = 14,
        grid = True,
        label = None,
        ax = rax
      )
      rax.hlines(
        xmin = minnode,
        xmax = maxnode,
        y = float(metrics[metrics["node"] != "C"][metric_name].mean()),
        color = mcolors.TABLEAU_COLORS["tab:red"],
        linewidth = 2
      )
      if ridx == 0:
        rax.set_title(key, fontsize = 14)
      if cidx == 0:
        rax.set_ylabel(metric_name, fontsize = 14)
      if ridx == nrows - 1:
        rax.set_xlabel("Node", fontsize = 14)
      ridx += 1
    cidx += 1
  if output_folder is not None:
    plt.savefig(
      os.path.join(
        output_folder, f"{'_'.join(relevant_metrics)}_{experiment_idx}.png"
      ),
      dpi = 300,
      format = "png",
      bbox_inches = "tight"
    )
    plt.close()
  else:
    plt.show()


def plot_predictions_with_average(
    all_predictions: pd.DataFrame, rolling_window: int = 4
  ):
  for (node, key), full_pred in all_predictions.groupby(["node", "key"]):
    pred = full_pred[["real", "pred"]].copy(deep = True)
    pred["real_avg"] = pred["real"].rolling(window = rolling_window).mean()
    pred.plot()
    plt.title(f"{node}, {key}")
    plt.show()
  


if __name__ == "__main__":
  # data_folder = "/Users/federicafilippini/Documents/GitHub/FORKs/gl-forecasting-edge/experiments/SERVER/seed4850/0"
  # nodes = list(range(9)) + ["centralized"]
  # all_predictions, all_metrics = load_existing_predictions(data_folder, nodes)
  # plot_predictions_with_average(all_predictions, 4)
  # plot_metrics(all_metrics, 0, ["mse", "mape"], data_folder)
  #
  base_folder = "/Users/federicafilippini/Documents/GitHub/FORKs/gl-forecasting-edge/experiments/SERVER/seed4850"
  for idx in range(1,10):
    all_predictions, all_metrics = compute_predictions(base_folder, idx)
    plot_metrics(
      all_metrics, 0, ["mse", "mape"], os.path.join(base_folder, "gossip")
    )
