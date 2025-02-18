from azurefunctions_utils import load_dataset

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import json
import os


data_folder = "/Users/federicafilippini/Documents/TEMP/gossip/t60/3layers/0"
nodes = list(range(9)) + ["centralized"]

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

for (node, key), full_pred in all_predictions.groupby(["node", "key"]):
  pred = full_pred[["real", "pred"]]
  pred["real_avg"] = pred["real"].rolling(window = 4).mean()
  pred.plot()
  plt.title(f"{node}, {key}")
  plt.show()

