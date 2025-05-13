from pathlib import Path
from typing import Callable

import keras
from keras.callbacks import ModelCheckpoint

from gossiplearning.config import Config
from utils.data import load_npz_data
from utils.evaluation import plot_node_history


def train_single_node(
    config: Config,
    datasets_folder: Path,
    output_folder: Path,
    model_creator: Callable[[], keras.Model],
    node: int,
    verbose: int = 0,
) -> None:
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_npz_data(
        str(datasets_folder / f"node_{node}.npz")
    )

    model = model_creator()

    # early_stopping = EarlyStopping(
    #     monitor="val_loss",
    #     patience=config.training.patience,
    #     min_delta=config.training.min_delta,
    #     restore_best_weights=True,
    # )

    plots_folder = datasets_folder / "plots" / "training"
    plots_folder.mkdir(exist_ok=True, parents=True)

    model_checkpoint = ModelCheckpoint(
        filepath=str(output_folder / f"{node}_single.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    history = model.fit(
        X_train,
        [Y_train[:, fn] for fn in range(config.training.n_output_vars)],
        validation_data=(
            X_val,
            [Y_val[:, fn] for fn in range(config.training.n_output_vars)],
        ),
        validation_batch_size=config.training.batch_size,
        verbose=verbose,
        callbacks=[
            # early_stopping,
            model_checkpoint,
        ],
        epochs=100,
        batch_size=config.training.batch_size,
        shuffle=config.training.shuffle_batch,
        use_multiprocessing=False,
    ).history

    for fn in range(config.training.n_output_vars):
        plot_node_history(
            history=history,
            file=plots_folder / f"node{node}_single.svg",
        )


def train_single_nodes(
    config: Config,
    datasets_folder: Path,
    output_folder: Path,
    model_creator: Callable[[], keras.Model],
    verbose: int = 0,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    for i in range(config.n_nodes):
        train_single_node(
            config,
            datasets_folder,
            output_folder,
            model_creator,
            i,
            verbose,
        )
