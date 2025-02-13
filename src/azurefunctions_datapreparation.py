from utils.data import encode_sequences_for_training, train_val_test_split
from utils.centralized_training import aggregate_datasets
from gossiplearning.config import Config, TrainingConfig

from azurefunctions_utils import encode_time, save_dataset

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
    "-i", "--data_folder", 
    help="Paths to the base data folder", 
    type=str,
    required=True
  )
  parser.add_argument(
    "-o", "--results_folder", 
    help="Paths to the base results folder", 
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
    "--simulation", 
    help="Simulation index", 
    default=0
  )
  args, _ = parser.parse_known_args()
  return args


def assign_nearest_tower(
    bbox_boundaries_file: str, taxis_df: pd.DataFrame, towers: pd.DataFrame
  ) -> pd.DataFrame:
  # define lat/lon boundaries
  LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = None, None, None, None
  with open(bbox_boundaries_file, "r") as f:
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX = eval(f.readline())
  LAT_DIFF = LAT_MAX - LAT_MIN
  LON_DIFF = LON_MAX - LON_MIN
  # number of cells (fixed) and cells coordinates
  N_ROWS = 20
  N_COLS = 20
  CELL_LON = LON_DIFF / N_COLS
  CELL_LAT = LAT_DIFF / N_ROWS
  # loop over taxi cells
  nearest_towers = []
  for cell in taxis_df["cell"]:
    i, j = divmod(cell, N_ROWS)
    lat_begin = LAT_MIN + i*CELL_LAT
    lon_begin = LON_MIN + j*CELL_LON
    lat_center = lat_begin + CELL_LAT / 2
    lon_center = lon_begin + CELL_LON / 2
    min_dist = None
    # identify nearest tower
    nearest_tower = None
    for n_tower, tower in towers.iterrows():
      dist = (lat_center - tower["lat"])**2 + (lon_center - tower["lon"])**2
      if min_dist is None or min_dist > dist:
        min_dist = dist
        nearest_tower = n_tower
    nearest_towers.append(nearest_tower)
  taxis_df["tower"] = nearest_towers
  return taxis_df


def build_nodes_dataframe(
    taxis_df: pd.DataFrame, all_data: pd.DataFrame
  ) -> pd.DataFrame:
  # define time boundaries
  day_min = int(all_data["day"].min())
  day_max = int(all_data["day"].max()) + 1
  hour_min = int(all_data["hour"].min())
  hour_max = int(all_data["hour"].max()) + 1
  # build dataframe
  nodes_dataset = {}
  for node, taxi_data in taxis_df.groupby("tower"):
    node_data = {"req": [], "day": [], "hour": []}
    for day in range(day_min, day_max):
      for hour in range(hour_min, hour_max):
        taxi_info = taxi_data[
          (
            taxi_data["day"] == day
          ) & (
            taxi_data["hour"] == hour
          )
        ]
        if len(taxi_info) > 0:
          # sum all requests in a given time window (+1 to avoid division by 0
          # when computing errors)
          all_req = int(
            all_data[
              (
                all_data["day"] == day
              ) & (
                all_data["hour"] == hour
              )
            ].dropna().set_index("tid").loc[taxi_info["taxi_id"]].sum()["req"]
          ) + 1
        else:
          # if there are no requests in a given time window, set 1 to avoid 
          # division by 0 when computing errors
          all_req = 1
        # add to dataset
        node_data["req"].append(all_req)
        node_data["day"].append(int(day))
        node_data["hour"].append(int(hour))
    # merge
    nodes_dataset[node] = pd.DataFrame(node_data)
  return nodes_dataset


def build_taxis_dataframe(
    full_taxi_path: pd.DataFrame, window: int, unit: str = "minute"
  ) -> pd.DataFrame:
  # boundaries
  seconds_in_day = 3600 * 24
  den = (60 if unit == "minute" else (3600 if unit == "hour" else 1))
  max_day = 14
  max_hour = int(seconds_in_day / window / den)
  # loop over days
  taxis_df = pd.DataFrame()
  for d in range(1, max_day + 1):
    # loop over time windows
    for h in range(1, max_hour + 1):
      # define boundaries
      sec_min = (h - 1) * window * den + (d - 1) * seconds_in_day
      sec_max = h * window * den + (d - 1) * seconds_in_day
      # print(f"d = {d}, h = {h}, min = {sec_min}, max = {sec_max}")
      # identify taxis that were around within the boundaries
      taxis = full_taxi_path[
        (
          full_taxi_path["second"] >= sec_min
        ) & (
          full_taxi_path["second"] < sec_max
        )
      ]
      # for each taxi, consider only the last occurrence
      minute_taxis = {"taxi_id": [], "cell": []}
      for taxi_id, info in taxis.groupby("taxi_id"):
        minute_taxis["taxi_id"].append(taxi_id)
        minute_taxis["cell"].append(int(info.iloc[-1]["cell"]))
      # concatenate
      minute_taxis = pd.DataFrame(minute_taxis)
      minute_taxis["hour"] = [h] * len(minute_taxis)
      minute_taxis["day"] = [d] * len(minute_taxis)
      taxis_df = pd.concat([taxis_df, minute_taxis], ignore_index = True)
  return taxis_df


def map_taxis_to_owners(
    all_taxis: list, function_owners: list, seed: int
  ) -> dict:
  # set seed
  np.random.seed(seed)
  # build mapping
  owner_taxi_mapping = {}
  for taxi_id in all_taxis:
    owner = np.random.choice(function_owners)
    owner_taxi_mapping[str(owner)] = int(taxi_id)
    function_owners.remove(owner)
  return owner_taxi_mapping


def plot_requests_by_day(
    singlenode_req: pd.DataFrame, output_folder: str, node: int
  ):
  colors = list(mcolors.TABLEAU_COLORS.values())[:7] * 2
  _, axs = plt.subplots(
    nrows = 1,
    ncols = 2,
    figsize = (20, 8)
  )
  for day, df1 in singlenode_req.groupby("day"):
    to_plot = df1.reset_index()
    to_plot.plot(
      x = "hour",
      y = "req",
      ax = axs[0],
      label = day,
      color = colors[day - 1]
    )
    to_plot["time"] = to_plot["hour"] + day * to_plot["hour"].max()
    to_plot.plot(
      x = "time",
      y = "req",
      ax = axs[1],
      label = day,
      color = colors[day - 1]
    )
  # save
  plt.savefig(
    os.path.join(output_folder, f"{node}_data.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )


def prepare_data(
    base_data_folder: str,
    t: int, 
    n: int, 
    k: int,
    seed: int,
    simulation: int,
    training_config: TrainingConfig,
    test_perc: float,
    val_perc_on_train: float,
    base_output_folder: str
  ):
  # build output folder
  output_folder = os.path.join(
    base_output_folder,
    f"azurefunctions-dataset2019/{n}n_k{k}_{t}min/seed{seed}/{simulation}"
  )
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    # load function traces
    print("Load functions traces")
    function_traces_file = os.path.join(
      base_data_folder, 
      f"azurefunctions-dataset2019/by_owner_flat_invocations_{t}min.csv"
    )
    all_data = pd.read_csv(function_traces_file)
    print("...done")
    # load taxi traces
    print("Load taxi traces")
    full_taxi_path = pd.read_csv(
      os.path.join(base_data_folder, "full_taxi_path.csv")
    )
    taxis_df = build_taxis_dataframe(full_taxi_path, t, "minute")
    print("...done")
    # load network and assign cells (thus, taxis) to nodes
    print("Load towers and assign cells to nodes")
    towers_file = os.path.join(
      base_data_folder, 
      f"networks/porto_{n}n_{k}k/seed{seed}/{simulation}/towers.csv"
    )
    towers = pd.read_csv(towers_file)
    taxis_df = assign_nearest_tower(
      "../assets/BBox_Porto.txt", taxis_df, towers
    )
    print("...done")
    # build taxi-owner identification
    print("Build taxi-owner identification")
    owner_taxi_mapping = map_taxis_to_owners(
      taxis_df["taxi_id"].unique(), list(all_data["fid"].unique()), seed
    )
    all_data["tid"] = [owner_taxi_mapping.get(o) for o in all_data["fid"]]
    print("...done")
    # build nodes dataset
    print("Build nodes dataset")
    nodes_dataset = build_nodes_dataframe(taxis_df, all_data)
    print("...done")
    # perform encoding and random train/val/test split
    print("Perform encoding and random train/val/test split")
    prepared_nodes_dataset = {}
    train_datasets = []
    val_datasets = []
    test_datasets = []
    n_auxiliary_features = None
    for node, node_data in nodes_dataset.items():
      print(f"    node {node}")
      # plot whole node dataset
      plot_requests_by_day(node_data, output_folder, node)
      # encode time
      time_encoded_data = encode_time(node_data)
      naf = len(time_encoded_data.columns) - 1
      if n_auxiliary_features is None:
        n_auxiliary_features = naf
      elif n_auxiliary_features != naf:
        raise RuntimeError(
          f"Inconsistent number of features: {n_auxiliary_features} != {naf}"
        )
      # encode sequences
      seq_encoded_X, seq_encoded_Y = encode_sequences_for_training(
        time_encoded_data, 
        training_config.input_timesteps, 
        training_config.output_timesteps, 
        training_config.n_output_vars, 
        n_auxiliary_features
      )
      # split
      X_train, Y_train, X_val, Y_val, X_test, Y_test = train_val_test_split(
        seq_encoded_X, 
        seq_encoded_Y, 
        test_perc = test_perc, 
        val_perc_on_train = val_perc_on_train
      )
      # save
      prepared_nodes_dataset[node] = (
        X_train, Y_train, X_val, Y_val, X_test, Y_test
      )
      # -- to file
      save_dataset(
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test,
        output_folder,
        node
      )
      # prepare datasets to train centralized model
      train_datasets.append((X_train, Y_train))
      val_datasets.append((X_val, Y_val))
      test_datasets.append((X_test, Y_test))
    print(f"...done; n_auxiliary_features = {n_auxiliary_features}")
    # aggregate and save centralized dataset
    print("Aggregate and save centralized dataset")
    centralized_train_data = aggregate_datasets(train_datasets)
    centralized_val_data = aggregate_datasets(val_datasets)
    centralized_test_data = aggregate_datasets(test_datasets)
    save_dataset(
      *centralized_train_data,
      *centralized_val_data,
      *centralized_test_data,
      output_folder,
      "centralized"
    )
    print("...done")
  else:
    print(f"WARNING: Output folder {output_folder} already exists!")
  return output_folder


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  base_data_folder = args.data_folder
  base_output_folder = args.results_folder
  config_file = args.config_file
  seed = args.seed
  simulation = args.simulation
  # load and validate configuration
  config = {}
  with open(config_file, "r") as f:
    config = Config.model_validate(json.load(f))
  # run
  prepare_data(
    base_data_folder, 
    t = config.data_preparation.time_window, 
    n = config.n_nodes, 
    k = config.connectivity, 
    seed = seed, 
    simulation = simulation, 
    training_config = config.training, 
    test_perc = config.data_preparation.test_perc, 
    val_perc_on_train = config.data_preparation.val_perc_on_train, 
    base_output_folder = base_output_folder
  )

