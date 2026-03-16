from category_encoders import BinaryEncoder
from typing import Tuple
import pandas as pd
import numpy as np
import json
import os


def encode_time(df: pd.DataFrame) -> pd.DataFrame:
  return BinaryEncoder(cols = ["hour", "day"]).fit_transform(df)


def load_dataset(
    data_folder: str, dataset_key: str
  ) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
  ]:
  X_train, Y_train, X_val, Y_val, X_test, Y_test = [None] * 6
  # open file
  filename = os.path.join(data_folder, f"{dataset_key}_data.json")
  with open(filename, "r") as ist:
    # load data
    data_json = json.load(ist)
    X_train = np.array(data_json["X_train"])
    Y_train = np.array(data_json["Y_train"])
    X_val = np.array(data_json["X_val"])
    Y_val = np.array(data_json["Y_val"])
    X_test = np.array(data_json["X_test"])
    Y_test = np.array(data_json["Y_test"])
  return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test)


def save_dataset(
    X_train: np.ndarray, 
    Y_train: np.ndarray, 
    X_val: np.ndarray, 
    Y_val: np.ndarray, 
    X_test: np.ndarray, 
    Y_test: np.ndarray, 
    output_folder: str,
    dataset_key: str
  ):
  # convert data to json
  data_json = {
    "X_train": X_train.tolist(), 
    "Y_train": Y_train.tolist(), 
    "X_val": X_val.tolist(), 
    "Y_val": Y_val.tolist(), 
    "X_test": X_test.tolist(), 
    "Y_test": Y_test.tolist()
  }
  # save
  filename = os.path.join(output_folder, f"{dataset_key}_data.json")
  with open(filename, "w") as ost:
    ost.write(json.dumps(data_json, indent = 2))

