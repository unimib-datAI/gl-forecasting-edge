from matplotlib import colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from re import sub
import parse
import json
import os

# Define a function to convert a string to camel case
def camel_case(s):
    # Use regular expression substitution to replace underscores and hyphens with spaces,
    # then title case the string (capitalize the first letter of each word), and remove spaces
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "\n")
    # Join the string, ensuring the first letter is lowercase
    return ''.join([s[0].upper(), s[1:]])

# Define a function to extract from a dataset the values associated to a 
# specific list of experiment indices
def extract_ids(data: pd.DataFrame, idxs: list) -> pd.DataFrame:
  filtered = pd.DataFrame()
  for idx in idxs:
    filtered = pd.concat(
      [filtered, data[data["idx"] == idx]], 
      ignore_index = True
    )
  return filtered

##############################################################################
# Load all values
##############################################################################
filename1 = "metrics.csv"
filename2 = "metrics_generalized.csv"
all_metrics = pd.DataFrame()
other_metrics = pd.DataFrame()
no_metrics_found = []
for foldername in os.listdir("."):
  if os.path.isdir(foldername) and foldername != "figures":
    file1 = os.path.join(foldername, filename1)
    file2 = os.path.join(foldername, filename2)
    if os.path.exists(file1) and os.path.exists(file2):
      idx = parse.parse("porto_{}", foldername)[0]
      # read metrics data
      metrics1 = pd.read_csv(os.path.join(foldername, filename1))
      metrics1["type"] = ["classical"] * len(metrics1)
      metrics1.rename(columns = {"Unnamed: 0": "method"}, inplace = True)
      metrics2 = pd.read_csv(os.path.join(foldername, filename2))
      metrics2["type"] = ["generalized"] * len(metrics2)
      metrics2.rename(columns = {"Unnamed: 0": "method"}, inplace = True)
      # read configuration file
      config = {}
      with open(os.path.join(foldername, "0", "config.json"), "r") as istream:
        config = json.load(istream)
      # combine information
      metrics = pd.concat([metrics1, metrics2], ignore_index = True)
      metrics["idx"] = [idx] * len(metrics)
      metrics["n_nodes"] = [config["n_nodes"]] * len(metrics)
      for k, v in config.get("training", {}).items():
        metrics[k] = [v] * len(metrics)
      # aggregate all info
      if "rmse" not in metrics.columns:
        all_metrics = pd.concat([all_metrics, metrics], ignore_index = True)
      else:
        other_metrics = pd.concat([other_metrics, metrics], ignore_index = True)
    else:
      no_metrics_found.append(foldername)

all_metrics.to_csv("all_metrics.csv", index = False)
other_metrics.to_csv("other_metrics.csv", index = False)

##############################################################################
# Load the configuration of experiments for which no metrics were computed
##############################################################################
no_metrics_config = pd.DataFrame()
for foldername in no_metrics_found:
  idx = parse.parse("porto_{}", foldername)[0]
  # read configuration file
  config = {}
  with open(os.path.join(foldername, "0", "config.json"), "r") as istream:
    config = json.load(istream)
  # extract and aggregate information
  config_df = {
    "idx": [int(idx)],
    "n_nodes": [config["n_nodes"]],
    **{k: [v] for k,v in config["training"].items()}
  }
  no_metrics_config = pd.concat(
    [no_metrics_config, pd.DataFrame(config_df)], ignore_index = True
  )
no_metrics_config.to_csv("no_metrics_config.csv", index = False)


##############################################################################
# Comparison with benchmark approaches
##############################################################################
comparison_with_benchmark_idxs = ["80", "82", "83", "84"]
comparison_metrics = pd.DataFrame()
for idx in comparison_with_benchmark_idxs:
  foldername = f"porto_{idx}"
  for exp_id in os.listdir(foldername):
    exp_folder = os.path.join(foldername, exp_id)
    if exp_id != "plots" and os.path.isdir(exp_folder):
      # load config to get information on the aggregation strategy
      strategy = None
      with open(os.path.join(exp_folder, "config.json"), "r") as istream:
        strategy = json.load(istream)["training"]["merge_strategy"]
      # load classical and generalized metrics
      metrics = pd.read_csv(
        os.path.join(exp_folder, "metrics.csv")
      ).rename(columns = {"Unnamed: 0": "method"}).set_index("method")
      gen_metrics = pd.read_csv(
        os.path.join(exp_folder, "generalization_metrics.csv")
      ).rename(columns = {"Unnamed: 0": "method"}).set_index("method")
      # join and add experiment id
      exp_metrics = metrics.join(
        gen_metrics, 
        on = "method", 
        lsuffix = "_classical", 
        rsuffix = "_generalized"
      )
      exp_metrics["foldername"] = [foldername] * len(exp_metrics)
      exp_metrics["exp_id"] = [exp_id] * len(exp_metrics)
      exp_metrics["method"] = exp_metrics.index
      exp_metrics["strategy"] = [strategy] * len(exp_metrics)
      exp_metrics.set_index("foldername", inplace = True)
      # merge
      comparison_metrics = pd.concat([comparison_metrics, exp_metrics])
comparison_metrics = comparison_metrics.replace(
  "improved_overwrite", "age_weighted_overwrite"
).replace(
  "age_weighted", "age_weighted_average"
).replace(
  "simple_avg", "simple_average"
)

###
### MMSE
###
aggregated = comparison_metrics[
  ["mse_classical", "mse_generalized", "method", "strategy"]
][
  comparison_metrics["method"].str.contains("aggregated")
]
aggregated["method"] = aggregated["method"].str.replace(" aggregated", "")
gossip = aggregated[aggregated["method"]=="gossip"].drop("method",axis=1)
gossip.index = range(len(gossip))
to_plot = pd.concat([
  gossip,
  aggregated[
    (aggregated["method"]=="single") & (aggregated["strategy"]=="overwrite")
  ].drop("strategy", axis = 1).rename(columns = {"method": "strategy"}),
  aggregated[
    (aggregated["method"]=="centralized") & (aggregated["strategy"]=="overwrite")
  ].drop("strategy", axis = 1).rename(columns = {"method": "strategy"})
], ignore_index = True)

to_plot.rename(columns = {"mse_classical": "MMSE"}, inplace = True)
to_plot["metric"] = ["classical"] * len(to_plot)
to_plot = pd.concat([
  to_plot.drop("mse_generalized", axis = 1),
  to_plot[["mse_generalized", "strategy", "metric"]].rename(
    columns = {"mse_generalized": "MMSE"}
  ).replace("classical", "generalized")
], ignore_index = True)
to_plot["strategy"] = to_plot["strategy"].apply(camel_case)

ax = sns.boxplot(
  data = to_plot, 
  x = "strategy", 
  y = "MMSE", 
  hue = "metric",
  showmeans = True,
  meanprops={"markerfacecolor": "r", "markeredgecolor": "r"}
)
plt.xlabel(None)
plt.ylabel("MMSE", fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 18)
plt.yscale("log")
plt.legend(fontsize = 18)
plt.grid(which = "both", axis = "y")
plt.show()


###
### MSE
###
not_aggregated = comparison_metrics[
  ["mse_classical", "mse_generalized", "method", "strategy"]
][
  ~comparison_metrics["method"].str.contains("aggregated")
]
not_aggregated["method"] = not_aggregated["method"].str.replace(" .*", "", regex = True)

gossip = not_aggregated[not_aggregated["method"]=="gossip"].drop("method",axis=1)
gossip.index = range(len(gossip))
to_plot = pd.concat([
  gossip,
  not_aggregated[
    (not_aggregated["method"]=="single") & (not_aggregated["strategy"]=="overwrite")
  ].drop("strategy", axis = 1).rename(columns = {"method": "strategy"}),
  not_aggregated[
    (not_aggregated["method"]=="centralized") & (not_aggregated["strategy"]=="overwrite")
  ].drop("strategy", axis = 1).rename(columns = {"method": "strategy"})
], ignore_index = True)

to_plot.rename(columns = {"mse_classical": "MSE"}, inplace = True)
to_plot["metric"] = ["classical"] * len(to_plot)
to_plot = pd.concat([
  to_plot.drop("mse_generalized", axis = 1),
  to_plot[["mse_generalized", "strategy", "metric"]].rename(
    columns = {"mse_generalized": "MSE"}
  ).replace("classical", "generalized")
], ignore_index = True)
to_plot["strategy"] = to_plot["strategy"].apply(camel_case)

ax = sns.boxplot(
  data = to_plot, 
  x = "strategy", 
  y = "MSE", 
  hue = "metric",
  palette="Blues",
  showmeans = True,
  meanprops={"markerfacecolor": "r", "markeredgecolor": "r"}
)
ax.axvline(
  x = 3.5,
  color = "k",
  linestyle = "dashed"
)
plt.xlabel(None)
plt.ylabel("MSE", fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 18)
plt.yscale("log")
plt.legend(fontsize = 18)
plt.grid(which = "both", axis = "y")
plt.show()




##############################################################################
# Change number of updates
##############################################################################
idx = "101"
foldername = f"porto_{idx}"
comparison_metrics = pd.DataFrame()
for exp_id in os.listdir(foldername):
  exp_folder = os.path.join(foldername, exp_id)
  if exp_id != "plots" and os.path.isdir(exp_folder):
    # load config to get information on the aggregation strategy
    strategy = None
    with open(os.path.join(exp_folder, "config.json"), "r") as istream:
      strategy = json.load(istream)["training"]["merge_strategy"]
    # load classical and generalized metrics
    metrics = pd.read_csv(
      os.path.join(exp_folder, "metrics.csv")
    ).rename(columns = {"Unnamed: 0": "method"}).set_index("method")
    gen_metrics = pd.read_csv(
      os.path.join(exp_folder, "generalization_metrics.csv")
    ).rename(columns = {"Unnamed: 0": "method"}).set_index("method")
    # join and add experiment id
    exp_metrics = metrics.join(
      gen_metrics, 
      on = "method", 
      lsuffix = "_classical", 
      rsuffix = "_generalized"
    )
    exp_metrics["foldername"] = [foldername] * len(exp_metrics)
    exp_metrics["exp_id"] = [exp_id] * len(exp_metrics)
    exp_metrics["method"] = exp_metrics.index
    exp_metrics["strategy"] = [strategy] * len(exp_metrics)
    exp_metrics.set_index("foldername", inplace = True)
    # merge
    comparison_metrics = pd.concat([comparison_metrics, exp_metrics])
comparison_metrics = comparison_metrics.replace(
  "age_weighted", "weighted_avg"
)

not_aggregated_gossip = comparison_metrics[
  ["mse_classical", "mse_generalized", "method", "strategy", "exp_id"]
][
  (
    ~comparison_metrics["method"].str.contains("aggregated")
  ) & (
    comparison_metrics["method"].str.contains("gossip")
  )
]

# not_aggregated_gossip.iloc[
#   not_aggregated_gossip["mse_generalized"].argmin()
# ][["exp_id", "method"]]

exp_id = "2"
node_id = "4"
not_aggregated_gossip[
  (
    not_aggregated_gossip["exp_id"] == exp_id
  ) & (
    not_aggregated_gossip["method"] == f"gossip {node_id}"
  )
]

node_training_history = {}
with open(os.path.join(foldername, exp_id, "history.json"), "r") as istream:
  node_training_history = json.load(istream)["nodes_training_history"][node_id]
node_training_history = pd.DataFrame(node_training_history)

node_training_history[["loss", "val_loss"]].plot()
plt.show()

node_training_history[["mse", "val_mse"]].plot(
  linewidth = 3,
  fontsize = 14
)
plt.xlabel("Epochs", fontsize = 18)
plt.ylabel("MSE", fontsize = 18)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.legend(["Training", "Validation"], fontsize = 18)
plt.grid()
plt.show()




##############################################################################
# Weights compression
##############################################################################
weights_compression_idxs = ["84", "87", "119", "121", "123"]
comparison_metrics = pd.DataFrame()
for idx in weights_compression_idxs:
  foldername = f"porto_{idx}"
  for exp_id in os.listdir(foldername):
    exp_folder = os.path.join(foldername, exp_id)
    if exp_id != "plots" and os.path.isdir(exp_folder):
      # load config to get information on the compression rate
      perc_sent_weights = None
      with open(os.path.join(exp_folder, "config.json"), "r") as istream:
        perc_sent_weights = json.load(istream)["training"]["perc_sent_weights"]
      # load classical and generalized metrics
      metrics = pd.read_csv(
        os.path.join(exp_folder, "metrics.csv")
      ).rename(columns = {"Unnamed: 0": "method"}).set_index("method")
      gen_metrics = pd.read_csv(
        os.path.join(exp_folder, "generalization_metrics.csv")
      ).rename(columns = {"Unnamed: 0": "method"}).set_index("method")
      # join and add experiment id
      exp_metrics = metrics.join(
        gen_metrics, 
        on = "method", 
        lsuffix = "_classical", 
        rsuffix = "_generalized"
      )
      exp_metrics["foldername"] = [foldername] * len(exp_metrics)
      exp_metrics["exp_id"] = [exp_id] * len(exp_metrics)
      exp_metrics["method"] = exp_metrics.index
      exp_metrics["rate"] = [perc_sent_weights] * len(exp_metrics)
      exp_metrics.set_index("foldername", inplace = True)
      # merge
      comparison_metrics = pd.concat([comparison_metrics, exp_metrics])
comparison_metrics = comparison_metrics.replace(
  "improved_overwrite", "weighted_overwrite"
).replace(
  "age_weighted", "weighted_avg"
)

###
### MSE
###
not_aggregated = comparison_metrics[
  ["mse_classical", "mse_generalized", "method", "rate"]
][
  ~comparison_metrics["method"].str.contains("aggregated")
]
not_aggregated["method"] = not_aggregated["method"].str.replace(" .*", "", regex = True)

to_plot = not_aggregated[not_aggregated["method"]=="gossip"].replace(
  "gossip", "weighted_avg"
)
to_plot["method"] = to_plot["method"].apply(camel_case)

to_plot.rename(columns = {"mse_classical": "MSE"}, inplace = True)
to_plot["metric"] = ["classical"] * len(to_plot)
to_plot = pd.concat([
  to_plot.drop("mse_generalized", axis = 1),
  to_plot[["mse_generalized", "method", "rate", "metric"]].rename(
    columns = {"mse_generalized": "MSE"}
  ).replace("classical", "generalized")
], ignore_index = True)

centralized = not_aggregated[
  (not_aggregated["method"]=="centralized") & (not_aggregated["rate"]==1.0)
]
c_classical = centralized[["mse_classical"]].rename(
  columns = {"mse_classical": "MSE"}
)
c_classical["rate"] = ["Centralized"] * len(centralized)
c_classical["metric"] = ["classical"] * len(centralized)
c_generalized = centralized[["mse_generalized"]].rename(
  columns = {"mse_generalized": "MSE"}
)
c_generalized["rate"] = ["Centralized"] * len(centralized)
c_generalized["metric"] = ["generalized"] * len(centralized)
#
single = not_aggregated[
  (not_aggregated["method"]=="single") & (not_aggregated["rate"]==1.0)
]
s_classical = single[["mse_classical"]].rename(
  columns = {"mse_classical": "MSE"}
)
s_classical["rate"] = ["Single"] * len(single)
s_classical["metric"] = ["classical"] * len(single)
s_generalized = single[["mse_generalized"]].rename(
  columns = {"mse_generalized": "MSE"}
)
s_generalized["rate"] = ["Single"] * len(single)
s_generalized["metric"] = ["generalized"] * len(single)

to_plot.sort_values(by = ["rate", "metric"], inplace = True)
to_plot = pd.concat([
  to_plot.drop("method", axis = 1),
  c_classical,
  c_generalized,
  s_classical,
  s_generalized
])

ax = sns.boxplot(
  data = to_plot, 
  x = "rate", 
  y = "MSE", 
  hue = "metric",
  palette="Blues",
  showmeans = True,
  meanprops={"markerfacecolor": "r", "markeredgecolor": "r"}
)
ax.axvline(
  x = 4.5,
  color = "k",
  linestyle = "dashed"
)
plt.xlabel(None)
plt.ylabel("MSE", fontsize = 20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 18)
plt.yscale("log")
plt.legend(fontsize = 18)
plt.grid(which = "both", axis = "y")
plt.show()


##############################################################################
# Number of messages varying the number of nodes
##############################################################################
no_metrics_config.sort_values(by = "idx", inplace = True)
no_metrics_config[
  ((
    no_metrics_config["fixed_updates"] == 25
  ) & (
    no_metrics_config["epochs_per_update"] == 4
  ) & (
    no_metrics_config["perc_sent_weights"] == 1.0
  )) & ((
    133 == no_metrics_config["idx"]
  ) | ((
    139 < no_metrics_config["idx"]
  ) & (
    no_metrics_config["idx"] <= 147
  )))
].transpose()

all_metrics[
  (all_metrics["n_nodes"] == 10) & (all_metrics["fixed_updates"] == 25)
].transpose()

by_n_nodes_idxs = [185] + list(range(140,148))
by_k_all_idxs = [185] + [
  idx for r, idx in enumerate(no_metrics_config["idx"]) if no_metrics_config.iloc[r]["fixed_updates"] == 25
]
by_updates_idxs = [146] + list(range(157,162))

all_messages_count = pd.DataFrame()
for idx in by_k_all_idxs:
  foldername = f"porto_{idx}"
  exp_row = None
  if idx != 185:
    exp_row = no_metrics_config[no_metrics_config["idx"]==idx]
  else:
    exp_row = all_metrics[all_metrics["idx"]==str(idx)]
  n_nodes = exp_row["n_nodes"].iloc[0]
  n_updates = exp_row["fixed_updates"].iloc[0]
  target_probability = exp_row["target_probability"].iloc[0]
  # loop over all experiments
  for exp_id in os.listdir(foldername):
    if os.path.isdir(os.path.join(foldername, exp_id)) and exp_id != "plots":
      # count the average number of links and transmission time
      n_edges = [0] * n_nodes
      wtt = [0] * n_nodes
      with open(os.path.join(foldername, exp_id, "config.json"), "r") as ist:
        nodes = json.load(ist)["nodes"]
        for node in nodes:
          n_edges[node["id"]] = len(node["links"])
          for link in node["links"]:
            wtt[node["id"]] += link["weights_transmission_time"]
          wtt[node["id"]] /= n_edges[node["id"]]
      # count the number of messages per node
      messages_sent = [0] * n_nodes
      messages_recv = [0] * n_nodes
      with open(os.path.join(foldername, exp_id, "history.json"), "r") as ist:
        history = json.load(ist)
        # increment the number of sent and received messages
        for message in history["messages"]:
          messages_sent[message["from_node"]] += 1
          messages_recv[message["to_node"]] += 1
      # average
      messages_count = {
        "sent": [sum(messages_sent) / n_nodes], 
        "recv": [sum(messages_recv) / n_nodes],
        "exp_id": [exp_id],
        "n_nodes": [n_nodes],
        "idx": [idx],
        "k": [sum(n_edges) / n_nodes],
        "weights_transmission_time": [sum(wtt) / n_nodes],
        "target_probability": [target_probability],
        "n_updates": [n_updates]
      }
      # add to the global list
      all_messages_count = pd.concat(
        [all_messages_count, pd.DataFrame(messages_count)], ignore_index = True
      )

##
## by_n_nodes
##
all_messages_avg = all_messages_count.groupby("n_nodes").mean(
  numeric_only = True
)
all_messages_avg["n_nodes"] = all_messages_avg.index

ax = all_messages_count.plot.scatter(
  x = "n_nodes",
  y = "sent",
  s = 30,
  fontsize = 14
)
all_messages_avg.plot.scatter(
  x = "n_nodes",
  y = "sent",
  ax = ax,
  c = "r",
  s = 40,
  linewidths = 3,
  marker = "x",
  grid = True
)
ax.set_xlabel("Number of nodes $n$", fontsize = 14)
ax.set_ylabel("Avg. number of messages per node $\\bar{m}$", fontsize = 14)
plt.show()

##
## by_k_all
##
msgs_by_k_all = all_messages_count[
  (
    all_messages_count["target_probability"] == 1.0
  ) & (
    all_messages_count["weights_transmission_time"] < 100
  )
]
o = mcolors.TABLEAU_COLORS["tab:orange"]
g = mcolors.TABLEAU_COLORS["tab:green"]
b = mcolors.TABLEAU_COLORS["tab:blue"]
colors = []
for i in msgs_by_k_all.index:
  row = msgs_by_k_all.loc[i]
  if row["idx"] in by_n_nodes_idxs:
    colors.append(b)
  elif row["n_nodes"] == 10:
    colors.append(o)
  elif row["n_nodes"] == 20:
    colors.append(g)
  else:
    colors.append("k")
msgs_by_k_all["color"] = colors

sns.regplot(
  data = msgs_by_k_all,
  x = "k",
  y = "sent",
  ci = None,
  scatter_kws = {
    "c": colors,
    "s": 30,
    "color": None
  },
  line_kws = {
    "color": "k",
    "linestyle": "dashed"
  }
)
plt.xlabel("Edge/Nodes ratio", fontsize = 14)
plt.ylabel("Avg. number of messages per node $\\bar{m}$", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(
  handles = [
    mpatches.Patch(color = o, label = "k-edge, 10-nodes network"),
    mpatches.Patch(color = g, label = "k-edge, 20-nodes network"),
    mpatches.Patch(color = b, label = "3-edge, n-nodes network")
  ],
  fontsize = 14
)
plt.grid()
plt.show()


##
## high_transfer_time
##
msgs_wtt = all_messages_count[
  (
    all_messages_count["n_nodes"] == 20
  )
].copy(deep = True)
colors = []
for i in msgs_wtt.index:
  row = msgs_wtt.loc[i]
  if row["target_probability"] == 0.25:
    colors.append(g)
  elif row["weights_transmission_time"] < 100:
    colors.append(b)
  else:
    colors.append(o)
msgs_wtt["color"] = colors
labels = {
  g: "Low neighbor sampling probability",
  b: "Normal transfer time",
  o: "High transfer time"
}

_, ax = plt.subplots()
for color, data in msgs_wtt.groupby("color"):
  sns.regplot(
    data = data,
    x = "k",
    y = "sent",
    ci = None,
    scatter_kws = {
      "s": 30
    },
    line_kws = {
      "linestyle": "dashed"
    },
    color = color,
    ax = ax,
    label = labels[color]
  )
plt.xlabel("Edge/Nodes ratio", fontsize = 14)
plt.ylabel("Avg. number of messages per node $\\bar{m}$", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.legend(
  # handles = [
  #   mpatches.Patch(color = o, label = "k-edge, 10-nodes network"),
  #   mpatches.Patch(color = g, label = "k-edge, 20-nodes network"),
  #   mpatches.Patch(color = b, label = "3-edge, n-nodes network")
  # ],
  fontsize = 14
)
plt.grid()
plt.show()

##
## by_updates
##
sns.regplot(
  data = all_messages_count,
  x = "n_updates",
  y = "sent",
  ci = None,
  scatter_kws = {
    "c": b,
    "s": 30,
    "color": None
  },
  line_kws = {
    "color": "k",
    "linestyle": "dashed"
  }
)
plt.xlabel("Number of model updates", fontsize = 14)
plt.ylabel("Avg. number of messages per node $\\bar{m}$", fontsize = 14)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.grid()
plt.show()


##
## centralized_comparison
##
centralized_comparison = all_messages_count[
  (
    all_messages_count["target_probability"] == 1.0
  ) & (
    all_messages_count["weights_transmission_time"] < 100
  ) & (
    (all_messages_count["k"] > 3) & (all_messages_count["k"] < 4)
  ) | (
    (all_messages_count["k"] > 5) & (all_messages_count["k"] < 7)
  )
].copy(deep = True)
centralized_comparison["k"] = [
  3 if 3 < k < 4 else 6 for k in centralized_comparison["k"]
]
centralized_comparison_avg = centralized_comparison.groupby(
  "k"
).mean(numeric_only = True)

max_n_nodes = 100
alphas = [
  (1.0, mcolors.CSS4_COLORS["lightgrey"]), 
  (2.0, mcolors.CSS4_COLORS["darkgrey"]), 
  (3.0, mcolors.CSS4_COLORS["grey"]), 
  (4.0, mcolors.CSS4_COLORS["dimgrey"]), 
  (4.7, mcolors.CSS4_COLORS["black"])
]
fig, ax = plt.subplots()
ax.plot(
  # k = 3
  range(max_n_nodes),
  [2 * centralized_comparison_avg.loc[3]["sent"]] * max_n_nodes,
  # k = 6
  range(max_n_nodes),
  [2 * centralized_comparison_avg.loc[6]["sent"]] * max_n_nodes,
  # style
  linestyle = "dashed",
  linewidth = 3
)
# centralized
for alpha, color in alphas:
  ax.plot(
    range(max_n_nodes),
    [2 * alpha * n for n in range(max_n_nodes)],
    linestyle = "solid",
    linewidth = 2,
    color = color
  )
ax.set_xlabel("Number of nodes $n$", fontsize = 18)
ax.set_ylabel("Network load", fontsize = 18)
ax.set_xticks(
  ax.get_xticks()[1:-1], 
  labels = [int(v) for v in ax.get_xticks()[1:-1]], 
  fontsize = 18
)
ax.set_yticks(
  ax.get_yticks()[1:-1], 
  labels = [f"{int(v)} M" for v in ax.get_yticks()[1:-1]], 
  fontsize = 16
)
ax.legend([
  "$\\overline{C}_{GL}$, k = 3",
  "$\\overline{C}_{GL}$, k = 6"
] + [
  "$C_{C}$" + f" ($\\alpha$ = {a})" for a, _ in alphas
], fontsize = 18)
plt.grid()
plt.show()


