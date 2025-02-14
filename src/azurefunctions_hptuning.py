import json
from pathlib import Path
from datetime import datetime

import keras
from keras import Sequential, Input
import keras_tuner
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dropout, Dense
from keras.metrics import RootMeanSquaredError
from keras.src.optimizers import Adam

from gossiplearning.config import Config

from azurefunctions_utils import load_dataset


def model_builder(hp: keras_tuner.HyperParameters) -> keras.Model:
    model = Sequential()
    # LSTM layer
    model.add(
        LSTM(
            hp.Int(
                min_value=16, max_value=320, step=16, name="lstm_layer_1_dim"
            ),
            activation="tanh",
            input_shape=(4, 14),
            return_sequences=True,
        )
    )
    # dropout
    if hp.Boolean("dropout_1"):
        model.add(
            Dropout(
                hp.Float(
                    name="dropout_1_rate", 
                    min_value=0.1, 
                    max_value=0.5, 
                    step=0.05
                )
            )
        )
    # LSTM layer
    if hp.Boolean("lstm_2"):
        # model.add(RepeatVector(1)),
        model.add(
            LSTM(
                hp.Int(
                    min_value=16, 
                    max_value=320, 
                    step=16, 
                    name="lstm_layer_2_dim"
                ),
                activation="tanh",
                input_shape=(4, 14),
                return_sequences=True,
            )
        )
    # dropout
    if hp.Boolean("dropout_2"):
        model.add(
            Dropout(
                hp.Float(
                    name="dropout_2_rate", 
                    min_value=0.1, 
                    max_value=0.5, 
                    step=0.05
                )
            )
        )
    # LSTM layer
    if hp.Boolean("lstm_3"):
        # model.add(RepeatVector(1)),
        model.add(
            LSTM(
                hp.Int(
                    min_value=16, 
                    max_value=320, 
                    step=16, 
                    name="lstm_layer_3_dim"
                ),
                activation="tanh",
                input_shape=(4, 14),
                # return_sequences=True,
            )
        )
    # dropout
    if hp.Boolean("dropout_3"):
        model.add(
            Dropout(
                hp.Float(
                    name="dropout_3_rate", 
                    min_value=0.1, 
                    max_value=0.5, 
                    step=0.05
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


if __name__ == "__main__":
    with open("azurefunctions_config.json", "r") as f:
      config = Config.model_validate(json.load(f))

    INPUT_SHAPE = (
      config.training.input_timesteps, config.training.n_input_features
    )

    max_n_trials = 100
    max_n_epochs = 100

    tuner = keras_tuner.RandomSearch(
        model_builder,
        objective='val_loss',
        max_trials=max_n_trials,
        directory="../experiments/hyperparameter_tuning",
        project_name=datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")
    )

    # train, val, _ = _build_centralized_model_dataset(
    #   Path("data/datasets/4func_10nodes/0"), config
    # )

    train, val, _ = load_dataset(
      "../experiments/azurefunctions-dataset2019/10n_k3_15min/seed4850/0", 
      "centralized"
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        restore_best_weights=True
    )

    tuner.search(
        train[0], 
        train[1], 
        validation_data=val, 
        callbacks=[early_stopping], 
        epochs=max_n_epochs
    )

    print(tuner.results_summary(num_trials=1))
