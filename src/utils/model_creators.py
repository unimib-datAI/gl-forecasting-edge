from keras.layers import Dropout
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

from gossiplearning.config import Config


def create_LSTM(config: Config) -> Model:
    optz = Adam(learning_rate=0.001, epsilon=1e-6)

    input_timesteps = 4

    inputs = Input(shape=(input_timesteps, config.training.n_input_features))

    lstm_layers = Sequential(
        [
            LSTM(
                50,
                activation="tanh",
                return_sequences=True,
            ),
            LSTM(
                50,
                activation="tanh",
                return_sequences=False,
            ),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dropout(0.2),
        ]
    )(inputs)

    outputs = [
        Dense(1, activation="relu", name=f"fn_{i}")(lstm_layers)
        for i in range(config.training.n_output_vars)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=optz,
        loss={f"fn_{i}": "mse" for i in range(config.training.n_output_vars)},
        metrics=["mae", "msle", "mse", "mape", RootMeanSquaredError()],
    )

    return model
