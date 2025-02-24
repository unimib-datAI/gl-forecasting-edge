from azurefunctions_train import compute_and_plot_predictions
from azurefunctions_utils import load_dataset

from matplotlib import colors as mcolors
from keras.api.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os


def compute_predictions(
    base_folder: str, simulation_index: int
  ) -> pd.DataFrame:
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
  for node, model in models.items():
    X_Y_data = all_X_Y_data[node]
    X_Y_data["common_test"] = [X_test, Y_test]
    compute_and_plot_predictions(
      X_Y_data, models[node], plot_folder, str(node)
    )


def load_existing_predictions(data_folder: str, nodes: list) -> pd.DataFrame:
  # loop over nodes
  all_predictions = pd.DataFrame()
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
      df = pd.DataFrame({
        "real": [float(y[0]) for y in Y_real],
        "pred": [float(y[0]) for y in pred],
        "key": [key] * len(Y_real),
        "node": [node] * len(Y_real)
      })
      all_predictions = pd.concat([all_predictions, df], ignore_index = True)
  return all_predictions


def plot_predictions_with_average(
    all_predictions: pd.DataFrame, rolling_window: int = 4
  ):
  for (node, key), full_pred in all_predictions.groupby(["node", "key"]):
    pred = full_pred[["real", "pred"]]
    pred["real_avg"] = pred["real"].rolling(window = rolling_window).mean()
    pred.plot()
    plt.title(f"{node}, {key}")
    plt.show()


if __name__ == "__main__":
  # data_folder = "/Users/federicafilippini/Documents/TEMP/gossip/t60/3layers/0"
  # nodes = list(range(9)) + ["centralized"]
  # all_predictions = load_existing_predictions(data_folder, nodes)
  # plot_predictions_with_average(all_predictions, 4)
  #
  models_folder = "../experiments/azurefunctions-dataset2019/10n_k3_15min/seed4850/gossip/0/models"
  common_test_file = "../experiments/azurefunctions-dataset2019/10n_k3_15min/seed4850/gossip/0/common_test_set.json"
  data_folder = "../experiments/azurefunctions-dataset2019/10n_k3_15min/seed4850/0"
  compute_predictions("../experiments/azurefunctions-dataset2019/10n_k3_15min/seed4850", 0)

