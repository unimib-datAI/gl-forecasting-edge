from gossiplearning.config import Config

from azurefunctions_utils import load_dataset

from keras.metrics import RootMeanSquaredError
from keras.layers import LSTM, Dropout, Dense
from keras.callbacks import EarlyStopping
from keras.src.optimizers import Adam
from datetime import datetime
from keras import Sequential
import keras_tuner
import argparse
import keras
import json
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(
    description="Run Hyperparameter Tuning"
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
    "--n_trials", 
    help="Maximum number of trials", 
    type=int,
    default=30
  )
  parser.add_argument(
    "--n_epochs", 
    help="Maximum number of epochs per trial", 
    type=int,
    default=100
  )
  args, _ = parser.parse_known_args()
  return args


def model_builder(hp: keras_tuner.HyperParameters) -> keras.Model:
  model = Sequential()
  # LSTM layer
  model.add(
    LSTM(
      hp.Int(min_value=16, max_value=320, step=16, name="lstm_layer_1_dim"),
      activation="tanh",
      input_shape=INPUT_SHAPE,
      return_sequences=True,
    )
  )
  # dropout
  if hp.Boolean("dropout_1"):
    model.add(
      Dropout(
        hp.Float(
          name="dropout_1_rate", min_value=0.1, max_value=0.5, step=0.05
        )
      )
    )
  # LSTM layer
  model.add(
    LSTM(
      hp.Int(min_value=16, max_value=320, step=16, name="lstm_layer_2_dim"),
      activation="tanh",
      input_shape=INPUT_SHAPE,
      return_sequences=True,
    )
  )
  # dropout
  if hp.Boolean("dropout_2"):
    model.add(
      Dropout(
        hp.Float(
          name="dropout_2_rate", min_value=0.1, max_value=0.5, step=0.05
        )
      )
    )
  # LSTM layer
  model.add(
    LSTM(
      hp.Int(min_value=16, max_value=320, step=16, name="lstm_layer_3_dim"),
      activation="tanh",
      input_shape=INPUT_SHAPE,
      return_sequences=False,
    )
  )
  # dropout
  if hp.Boolean("dropout_3"):
    model.add(
      Dropout(
        hp.Float(
          name="dropout_3_rate", min_value=0.1, max_value=0.5, step=0.05
        )
      )
    )
  # Dense layer
  if hp.Boolean("dense"):
    model.add(
      Dense(
        hp.Int(min_value=16, max_value=320, step=16, name="dense_dim"),
        activation="relu"
      )
    )
  # dropout
  if hp.Boolean("dropout_4"):
    model.add(
      Dropout(
        hp.Float(
          name="dropout_4_rate", min_value=0.1, max_value=0.5, step=0.05
        )
      )
    )
  # output layer
  model.add(
    Dense(1, activation='relu')
  )
  # optimizer
  optz = Adam(
    learning_rate=hp.Float(
      min_value=0.0005, max_value=0.01, step=0.0005, name="learning_rate"
    )
  )
  # compile model
  model.compile(
    optimizer=optz, 
    loss='mse', 
    metrics=["mae", "msle", "mse", "mape", RootMeanSquaredError()]
  )
  return model


def run(
    config_file: str,
    base_folder: str,
    seed: int,
    max_n_trials: int,
    max_n_epochs: int,
    simulation: int
  ):
  # load and validate configuration
  config = None
  with open(config_file, "r") as f:
    config = Config.model_validate(json.load(f))
  # define NN input shape
  global INPUT_SHAPE
  INPUT_SHAPE = (
    config.training.input_timesteps, config.training.n_input_features
  )
  # define tuner
  hp_directory = os.path.join(base_folder, "hyperparameter_tuning")
  project_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
  tuner = keras_tuner.RandomSearch(
    model_builder,
    objective='val_loss',
    max_trials=max_n_trials,
    directory=hp_directory,
    project_name=project_name
  )
  # load (centralized) dataset
  n = config.n_nodes
  k = config.connectivity
  t = config.data_preparation.time_window
  train, val, _ = load_dataset(
    os.path.join(
      base_folder,
      f"azurefunctions-dataset2019/{n}n_{k}k_{t}min/seed{seed}/{simulation}"
    ), 
    "centralized"
  )
  # define callback for early stopping
  early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=config.training.patience,
    min_delta=config.training.min_delta,
    restore_best_weights=True
  )
  # run
  tuner.search(
    train[0], 
    train[1], 
    validation_data=val, 
    callbacks=[early_stopping], 
    epochs=max_n_epochs
  )
  # print result
  result_file = os.path.join(hp_directory, project_name, "result.txt")
  with open(result_file, "w") as ostream:
    ostream.write(tuner.results_summary(num_trials=1))


if __name__ == "__main__":
  # parse arguments
  args = parse_arguments()
  config_file = args.config_file
  base_folder = args.base_folder
  seed = args.seed
  max_n_trials = args.n_trials
  max_n_epochs = args.n_epochs
  # run
  run(
    config_file=config_file,
    base_folder=base_folder,
    seed=seed,
    max_n_trials=max_n_trials,
    max_n_epochs=max_n_epochs,
    simulation=0
  )
  