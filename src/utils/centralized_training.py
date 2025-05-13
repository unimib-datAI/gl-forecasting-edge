from pathlib import Path
from typing import Callable

import keras
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle

from gossiplearning.config import Config
from gossiplearning.models import LabelledData
from utils.data import load_npz_data
from utils.evaluation import plot_node_history


def aggregate_datasets(datasets: list[LabelledData]) -> LabelledData:
    X = np.concatenate([X for X, Y in datasets], axis=0)
    Y = np.concatenate([Y for Y, Y in datasets], axis=0)
    return shuffle(X, Y)


def _train_centralized(
    train: LabelledData,
    validation: LabelledData,
    config: Config,
    model_output_path: Path,
    model_creator: Callable[[], keras.Model],
    plots_folder: Path,
    verbose: int,
) -> None:
    model = model_creator()

    model_output_path.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        restore_best_weights=True,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=str(model_output_path / f"centralized.h5"),
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )
    history = model.fit(
        train[0],
        [train[1][:, fn] for fn in range(config.training.n_output_vars)],
        validation_data=(
            validation[0],
            [validation[1][:, fn] for fn in range(config.training.n_output_vars)],
        ),
        verbose=verbose,
        callbacks=[
            # early_stopping,
            model_checkpoint,
        ],
        epochs=100,
        batch_size=config.training.batch_size,
        use_multiprocessing=True,
        shuffle=config.training.shuffle_batch,
    ).history

    for fn in range(config.training.n_output_vars):
        plot_node_history(
            history=history,
            file=plots_folder / f"fn_{fn}_centralized.svg",
        )


def _build_centralized_model_dataset(
    node_datasets_folder: Path,
    config: Config,
) -> tuple[LabelledData, LabelledData, LabelledData]:
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for i in range(config.n_nodes):
        X_train, Y_train, X_val, Y_val, X_test, Y_test = load_npz_data(
            str(node_datasets_folder / f"node_{i}.npz")
        )

        train_datasets.append((X_train, Y_train))
        val_datasets.append((X_val, Y_val))
        test_datasets.append((X_test, Y_test))

    return (
        aggregate_datasets(train_datasets),
        aggregate_datasets(val_datasets),
        aggregate_datasets(test_datasets),
    )


def train_centralized_model(
    node_datasets_folder: Path,
    config: Config,
    model_output_path: Path,
    model_creator: Callable[[], keras.Model],
    verbose: int = 0,
) -> None:
    plots_folder = node_datasets_folder / "plots" / "training"
    plots_folder.mkdir(exist_ok=True, parents=True)

    model_output_path.mkdir(exist_ok=True, parents=True)

    train, val, _ = _build_centralized_model_dataset(node_datasets_folder, config)
    _train_centralized(
        train,
        val,
        config,
        model_output_path,
        model_creator,
        plots_folder,
        verbose,
    )
