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
    models: dict, all_X_Y_data: dict, plot_folder: str
  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # compute and plot predictions
  all_predictions = pd.DataFrame()
  all_metrics = {"key": [], "node": [], "metrics": []}
  for node, model in models.items():
    X_Y_data = all_X_Y_data[node]
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


def load_gossip_models_and_data(
    base_folder: str, simulation_index: int, merge_strategy: str
  ) -> Tuple[dict, dict, str]:
  models_folder = os.path.join(
    base_folder, 
    f"gossip-MergeStrategy.{merge_strategy}", 
    str(simulation_index), 
    "models"
  )
  data_folder = os.path.join(
    base_folder, str(simulation_index)
  )
  common_test_file = os.path.join(
    base_folder, 
    f"gossip-MergeStrategy.{merge_strategy}", 
    str(simulation_index), 
    "common_test_set.json"
  )
  plot_folder = os.path.join(
    base_folder, 
    f"gossip-MergeStrategy.{merge_strategy}", 
    str(simulation_index), 
    "plots"
  )
  # load common test set
  X_test, Y_test = None, None
  with open(common_test_file, "r") as istream:
    common_test_set = json.load(istream)
    X_test = np.array(common_test_set["X_test"])
    Y_test = np.array(common_test_set["Y_test"])
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
      "test": test,
      "common_test": [X_test, Y_test]
    }
  return models, all_X_Y_data, plot_folder


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


def load_single_centralized_models(data_folder: str, nodes: list) -> dict:
  models = {}
  for node in nodes:
    # get model filename
    model_filename = None
    if str(node) != "centralized":
      model_filename = os.path.join(
        data_folder, "results", f"{node}_single.keras"
      )
    else:
      model_filename = os.path.join(
        data_folder, "results", "centralized.keras"
      )
    # load model
    model = load_model(model_filename)
    models[node] = model
  return models


def load_single_data(data_folder: str, nodes: list) -> dict:
  single_X_Y_data = {}
  for node in nodes:
    _, _, test = load_dataset(data_folder, node)
    single_X_Y_data[node] = {"test": test}
  return single_X_Y_data


def plot_metrics(
    all_metrics: pd.DataFrame, 
    experiment_idx: int, 
    relevant_metrics: list,
    output_folder: str = None
  ):
  # rename 'centralized' to 'C' (if available)
  if (all_metrics["node"] == "centralized").any():
    all_metrics.replace("centralized", "C", inplace = True)
  # get maximum and minimum number of nodes
  minnode = min([int(n) for n in all_metrics["node"].unique() if n != "C"])
  maxnode = max([int(n) for n in all_metrics["node"].unique() if n != "C"])
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
    all_predictions: pd.DataFrame, 
    rolling_window: int = 4, 
    output_folder: str = None
  ):
  for (node, key), full_pred in all_predictions.groupby(["node", "key"]):
    pred = full_pred[["real", "pred"]].copy(deep = True)
    pred["real_avg"] = pred["real"].rolling(window = rolling_window).mean()
    pred.plot()
    plt.title(f"{node}, {key}")
    if output_folder is not None:
      plt.savefig(
        os.path.join(
          output_folder, f"predictions_with_average_{node}_{key}.png"
        ),
        dpi = 300,
        format = "png",
        bbox_inches = "tight"
      )
      plt.close()
    else:
      plt.show()
  


if __name__ == "__main__":
  base_folder = "/Users/federicafilippini/Documents/GitHub/FORKs/gl-forecasting-edge/experiments/SERVER/10n_3k_15min/seed1000"
  merge_strategy = "AGE_WEIGHTED"
  nodes = list(range(9)) + ["centralized"]
  all_test_avg_metrics = pd.DataFrame()
  for idx in range(0,10):
    # single/centralized predictions
    data_folder = os.path.join(
      base_folder, str(idx)
    )
    sc_predictions, sc_metrics = load_existing_predictions(data_folder, nodes)
    plot_predictions_with_average(sc_predictions, 4, data_folder)
    plot_metrics(sc_metrics, idx, ["mse", "mape"], data_folder)
    # gossip predictions and metrics
    gossip_models, gossip_X_Y_data, gplot_folder = load_gossip_models_and_data(
      base_folder, idx, merge_strategy
    )
    gossip_predictions, gossip_metrics = compute_predictions(
      gossip_models, gossip_X_Y_data, gplot_folder
    )
    plot_predictions_with_average(
      gossip_predictions, 
      4, 
      os.path.join(base_folder, f"gossip-MergeStrategy.{merge_strategy}")
    )
    plot_metrics(
      gossip_metrics, 
      idx, 
      ["mse", "mape"], 
      os.path.join(base_folder, f"gossip-MergeStrategy.{merge_strategy}")
    )
    # compute common-test predictions with single/centralized models
    common_test = {"common_test": gossip_X_Y_data[0]["common_test"]}
    sc_models = load_single_centralized_models(data_folder, nodes)
    #
    plot_folder = os.path.join(data_folder, "common_test_predictions")
    os.makedirs(plot_folder, exist_ok = True)
    sc_generalized_metrics = {"node": [], "metrics": []}
    for node, model in sc_models.items():
      predictions = compute_and_plot_predictions(
        common_test, model, plot_folder, str(node)
      )
      metrics = compute_metrics(
        common_test["common_test"][1], predictions["common_test"]
      )
      sc_generalized_metrics["node"].append(node)
      sc_generalized_metrics["metrics"].append(metrics)
    sc_generalized_metrics = pd.DataFrame(sc_generalized_metrics)
    sc_generalized_metrics["mape"] = [
      m.mape * 100 for m in sc_generalized_metrics["metrics"]
    ]
    sc_generalized_metrics["mse"] = [
      m.mse for m in sc_generalized_metrics["metrics"]
    ]
    # compute local predictions with centralized models
    local_X_Y_data = load_single_data(data_folder, nodes)
    plot_folder = os.path.join(data_folder, "local_centralized_predictions")
    os.makedirs(plot_folder, exist_ok = True)
    c_local_metrics = {"node": [], "metrics": []}
    for node, X_Y_data in local_X_Y_data.items():
      pred = compute_and_plot_predictions(
        X_Y_data, sc_models["centralized"], plot_folder, node
      )
      metrics = compute_metrics(X_Y_data["test"][1], pred["test"])
      c_local_metrics["node"].append(node)
      c_local_metrics["metrics"].append(metrics)
    c_local_metrics = pd.DataFrame(c_local_metrics)
    c_local_metrics["mape"] = [
      m.mape * 100 for m in c_local_metrics["metrics"]
    ]
    c_local_metrics["mse"] = [
      m.mse for m in c_local_metrics["metrics"]
    ]
    # save test metrics
    test_avg_metrics = pd.concat(
      [
        # -- single
        pd.DataFrame(
          sc_metrics[
            (
              sc_metrics["key"] == "test"
            ) & (
              sc_metrics["node"] != "centralized"
            )
          ].mean(numeric_only = True),
          columns = ["single"]
        ).transpose(),
        # -- centralized (on local data)
        pd.DataFrame(
          c_local_metrics[
            c_local_metrics["node"] != "centralized"
          ][["mape", "mse"]].mean(numeric_only = True),
          columns = ["centralized"]
        ).transpose(),
        # -- gossip
        pd.DataFrame(
          gossip_metrics[gossip_metrics["key"] == "test"].mean(
            numeric_only = True
          ),
          columns = ["gossip"]
        ).transpose().drop("node", axis = "columns"),
        # -- single (generalized)
        pd.DataFrame(
          sc_generalized_metrics[
            sc_generalized_metrics["node"] != "centralized"
          ].mean(numeric_only = True),
          columns = ["single (generalized)"]
        ).transpose(),
        # -- centralized (generalized)
        sc_generalized_metrics[
          sc_generalized_metrics["node"] == "centralized"
        ][["node", "mape", "mse"]].replace(
          "centralized", "centralized (generalized)"
        ).set_index("node", drop = True),
        # -- gossip (generalized)
        pd.DataFrame(
          gossip_metrics[gossip_metrics["key"] == "common_test"].mean(
            numeric_only = True
          ),
          columns = ["gossip (generalized)"]
        ).transpose().drop("node", axis = "columns")
      ]
    )
    test_avg_metrics["idx"] = [idx] * len(test_avg_metrics)
    all_test_avg_metrics = pd.concat(
      [all_test_avg_metrics, test_avg_metrics]
    )
  all_test_avg_metrics.to_csv(
    os.path.join(
      base_folder, f"all_test_avg_metrics-MergeStrategy.{merge_strategy}.csv"
    )
  )
