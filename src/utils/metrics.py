import math
from pathlib import Path
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pydantic import BaseModel
from sklearn import metrics


class Metrics(BaseModel):
    """Class wrapper for a common set of regression metrics."""

    rmse: float
    mae: float
    mape: float
    msle: float
    mse: float


class SimulationNodeMetrics(BaseModel):
    gossip: list[Metrics]
    single: list[Metrics]
    centralized: list[Metrics]
    gossip_generalized: list[Metrics]
    single_generalized: list[Metrics]
    centralized_generalized: list[Metrics]


class SimulationMetrics(BaseModel):
    """Wrapper class for simulation result metrics."""

    gossip: Metrics
    single_training: Metrics
    centralized: Metrics


def compute_metrics(
    truth: np.ndarray, predicted: Union[np.ndarray, Sequence[np.ndarray]]
) -> Metrics:
    flattened_pred = np.concatenate(predicted)
    n_elements = math.prod(truth.shape)
    flattened_truth = truth.reshape((n_elements,), order="F")

    return Metrics(
        mse=metrics.mean_squared_error(flattened_truth, flattened_pred),
        rmse=math.sqrt(metrics.mean_squared_error(flattened_truth, flattened_pred)),
        msle=metrics.mean_squared_log_error(
            flattened_truth, np.clip(flattened_pred, a_min=0, a_max=None)
        ),
        mae=metrics.mean_absolute_error(flattened_truth, flattened_pred),
        mape=metrics.mean_absolute_percentage_error(flattened_truth, flattened_pred),
    )


def compute_generalized_metrics(
    truth: Sequence[np.ndarray],
    predicted: Union[np.ndarray, Sequence[np.ndarray]],
) -> Metrics:
    flattened_pred = np.concatenate([np.concatenate(p) for p in predicted])
    flattened_truth = np.concatenate(
        [t.reshape((math.prod(t.shape),), order="F") for t in truth]
    )

    return Metrics(
        mse=metrics.mean_squared_error(flattened_truth, flattened_pred),
        rmse=math.sqrt(metrics.mean_squared_error(flattened_truth, flattened_pred)),
        msle=metrics.mean_squared_log_error(
            flattened_truth, np.clip(flattened_pred, a_min=0, a_max=None)
        ),
        mae=metrics.mean_absolute_error(flattened_truth, flattened_pred),
        mape=metrics.mean_absolute_percentage_error(flattened_truth, flattened_pred),
    )


def average_metrics(metrics: Sequence[Metrics]) -> Metrics:
    n = len(metrics)

    return Metrics(
        mae=sum([m.mae for m in metrics]) / n,
        rmse=sum([m.rmse for m in metrics]) / n,
        mape=sum([m.mape for m in metrics]) / n,
        mse=sum([m.mse for m in metrics]) / n,
        msle=sum([m.msle for m in metrics]) / n,
    )


def dump_metrics(
    gossip_metrics: Sequence[Metrics],
    single_metrics: Sequence[Metrics],
    centralized_metrics: Sequence[Metrics],
    aggregated: SimulationMetrics,
    file: Path,
) -> None:
    df = pd.DataFrame(
        data=[
            aggregated.gossip.model_dump(),
            aggregated.single_training.model_dump(),
            aggregated.centralized.model_dump(),
        ],
        index=["gossip aggregated", "single aggregated", "centralized aggregated"],
    )

    for i, node_metrics in enumerate(gossip_metrics):
        df = pd.concat([df, pd.DataFrame(node_metrics.dict(), index=[f"gossip {i}"])])

    for i, node_metrics in enumerate(single_metrics):
        df = pd.concat([df, pd.DataFrame(node_metrics.dict(), index=[f"single {i}"])])

    for i, node_metrics in enumerate(centralized_metrics):
        df = pd.concat(
            [df, pd.DataFrame(node_metrics.dict(), index=[f"centralized {i}"])]
        )

    df.to_csv(file)


def dump_experiment_metrics(
    gossip_metrics: Sequence[Metrics],
    single_metrics: Sequence[Metrics],
    centralized: Sequence[Metrics],
    file: Path,
) -> None:
    metric_names = ["mse", "mae", "rmse", "mae"]

    def _aggregate(metrics: Sequence[Metrics]) -> dict[str, float]:
        return {
            "mse_mean": np.average([m.mse for m in metrics]),
            "mse_avg": np.var([m.mse for m in metrics]),
            "mse_std": np.std([m.mse for m in metrics]),
            "mse_50q": np.percentile([m.mse for m in metrics], q=50),
            "mse_90q": np.percentile([m.mse for m in metrics], q=90),
            "rsd": np.std([m.mse for m in metrics])
            / np.average([m.mse for m in metrics]),
        }

    df = pd.DataFrame(
        data=[
            _aggregate(gossip_metrics),
            _aggregate(single_metrics),
            _aggregate(centralized),
        ],
        index=["Gossip", "Single", "Centralized"],
        columns=[
            "mse_mean",
            "mse_var",
            "mse_std",
            "mse_50q",
            "mse_90q",
            "rsd",
        ],
    )

    df.to_csv(file)


def load_experiment_metrics(file: Path) -> SimulationMetrics:
    metrics_dict = pd.read_csv(file, header=0, index_col=0).transpose().to_dict()

    return SimulationMetrics(
        gossip=Metrics.model_validate(metrics_dict["Gossip"]),
        single_training=Metrics.model_validate(metrics_dict["Single"]),
        centralized=Metrics.model_validate(metrics_dict["Centralized"]),
    )


def plot_node_metrics(
    gossip_nodes: list[Metrics],
    single_nodes: list[Metrics],
    centralized_nodes: list[Metrics],
    aggregated: SimulationMetrics,
    file: Path,
) -> None:
    width = 0.25
    n_nodes = len(gossip_nodes)
    x_ticks = np.arange(n_nodes + 1)
    center = width
    metrics = ["rmse", "mse", "msle", "mae"]

    plt.ioff()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    gossip_metrics_arrays = {
        metric: np.array(
            [getattr(aggregated.gossip, metric)]
            + [getattr(gossip_nodes[i], metric) for i in range(n_nodes)]
        )
        for metric in metrics
    }

    single_metrics_arrays = {
        metric: np.array(
            [getattr(aggregated.single_training, metric)]
            + [getattr(single_nodes[i], metric) for i in range(n_nodes)]
        )
        for metric in metrics
    }

    centralized_metrics_arrays = {
        metric: np.array(
            [getattr(aggregated.centralized, metric)]
            + [getattr(centralized_nodes[i], metric) for i in range(n_nodes)]
        )
        for metric in metrics
    }

    for i, metric in enumerate(metrics):
        row = int(math.floor(i / 2))
        col = i % 2

        axes[row, col].bar(
            x_ticks, gossip_metrics_arrays[metric], width=width, label="Gossip"
        )
        axes[row, col].bar(
            x_ticks + width, single_metrics_arrays[metric], width=width, label="Single"
        )
        axes[row, col].bar(
            x_ticks + 2 * width,
            centralized_metrics_arrays[metric],
            width=width,
            label="Centralized",
        )
        axes[row, col].set_title(f"{metric.upper()}")
        axes[row, col].set_xticks(
            x_ticks + center, ["Global"] + [f"Node {i}" for i in range(n_nodes)]
        )
        axes[row, col].legend(loc="upper left")

    plt.savefig(file, dpi=500)
    plt.close()


def plot_metrics(metrics: SimulationMetrics, file: Path) -> None:
    width = 0.8
    x_ticks = ["Gossip", "Single", "Centralized"]
    color = ["tomato", "royalblue", "darkseagreen"]
    metric_names = ["rmse", "mse", "msle", "mae"]

    plt.ioff()
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    for i, metric in enumerate(metric_names):
        row = int(math.floor(i / 2))
        col = i % 2

        axes[row, col].bar(
            x_ticks,
            np.array(
                [
                    getattr(metrics.gossip, metric),
                    getattr(metrics.single_training, metric),
                    getattr(metrics.centralized, metric),
                ]
            ),
            width=width,
            color=color,
        )
        axes[row, col].set_title(metric.upper())

    plt.savefig(file, dpi=500)
    plt.close()


def plot_predicted_time_series(
    n_functions: int,
    gossip_predictions: Sequence[np.ndarray],
    single_predictions: Sequence[np.ndarray],
    centralized_predictions: Sequence[np.ndarray],
    truth: np.ndarray,
    file: Path,
    begin_at: Optional[int] = None,
    end_at: Optional[int] = None,
) -> None:
    rows, cols = _find_optimal_display(n_functions)

    plt.ioff()
    fig, axes = plt.subplots(rows, cols, figsize=(18, 12))

    for i in range(n_functions):
        row = int(math.floor(i / cols))
        col = i % cols
        ax = axes if (rows == 1 and cols == 1) else axes[row, col]

        if n_functions > 1:
            flat_gossip = gossip_predictions[i].flatten()
            flat_single = single_predictions[i].flatten()
            flat_centralized = centralized_predictions[i].flatten()
            flat_truth = truth[:, i].flatten()
        else:
            flat_gossip = gossip_predictions.flatten()
            flat_single = single_predictions.flatten()
            flat_centralized = centralized_predictions.flatten()
            flat_truth = truth.flatten()

        ax.plot(
            flat_truth[begin_at:end_at],
            label="Truth",
            color="royalblue",
            linewidth=1,
        )
        ax.plot(
            flat_gossip[begin_at:end_at],
            label="Gossip predictions",
            color="tomato",
            linewidth=1,
        )
        ax.plot(
            flat_centralized[begin_at:end_at],
            label="Centralized predictions",
            color="orchid",
            linewidth=1,
        )
        ax.plot(
            flat_single[begin_at:end_at],
            label="Single predictions",
            color="darkseagreen",
            linewidth=1,
        )

        ax.legend()
        ax.set_title(f"Function {i + 1}")

    plt.savefig(file, format="svg", dpi=500)
    plt.close()


def plot_metrics_violinplot(
    gossip: Sequence[Metrics],
    single: Sequence[Metrics],
    centralized: Sequence[Metrics],
    folder: Path,
    file_prefix: str | None = None,
) -> None:
    metric_names = ["rmse", "mse", "msle", "mae"]
    plot_data = {
        metric: {
            "Gossip": [getattr(m, metric) for m in gossip],
            "Offline": [getattr(m, metric) for m in single],
            "Centralized": [getattr(m, metric) for m in centralized],
        }
        for metric in metric_names
    }

    plt.ioff()

    for i, metric in enumerate(metric_names):
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        sns.boxplot(data=plot_data[metric], whis=[0, 95], ax=ax)
        ax.grid(visible=True, axis="y")
        ax.set_ylabel(metric.upper())

        file_name = (
            f"{file_prefix}_{metric}.svg"
            if file_prefix is not None
            else f"{metric}.svg"
        )

        plt.savefig(folder / file_name, format="svg", dpi=500, bbox_inches="tight")
        plt.close()


def _find_optimal_display(n: int) -> tuple[int, int]:
    # find best number of rows and columns depending on the number of subplots to display
    options = []

    max_rows = int(math.ceil(n / 2))

    for rows in range(1, max_rows + 1):
        columns = math.ceil(n / rows)
        diff = abs(rows - columns)

        options.append((rows, columns, diff))

    best = min(options, key=lambda x: x[2])

    return best[0], best[1]


def load_metrics(
    exp_path: Path, n_sim: int, n_nodes: int
) -> tuple[list[SimulationMetrics], list[SimulationMetrics]]:
    sim_metrics = []
    gen_metrics = []

    for j in range(n_sim):
        metrics = pd.read_csv(exp_path / str(j) / "metrics.csv")
        gossip = [
            Metrics(
                rmse=metrics.iloc[i + 3]["rmse"],
                mae=metrics.iloc[i + 3]["mae"],
                mape=metrics.iloc[i + 3]["mape"],
                msle=metrics.iloc[i + 3]["msle"],
                mse=metrics.iloc[i + 3]["mse"],
            )
            for i in range(n_nodes)
        ]
        single = [
            Metrics(
                rmse=metrics.iloc[i + n_nodes + 3]["rmse"],
                mae=metrics.iloc[i + n_nodes + 3]["mae"],
                mape=metrics.iloc[i + n_nodes + 3]["mape"],
                msle=metrics.iloc[i + n_nodes + 3]["msle"],
                mse=metrics.iloc[i + n_nodes + 3]["mse"],
            )
            for i in range(n_nodes)
        ]
        centralized = [
            Metrics(
                rmse=metrics.iloc[i + 2 * n_nodes + 3]["rmse"],
                mae=metrics.iloc[i + 2 * n_nodes + 3]["mae"],
                mape=metrics.iloc[i + 2 * n_nodes + 3]["mape"],
                msle=metrics.iloc[i + 2 * n_nodes + 3]["msle"],
                mse=metrics.iloc[i + 2 * n_nodes + 3]["mse"],
            )
            for i in range(n_nodes)
        ]
        sm = SimulationMetrics(
            gossip=average_metrics(gossip),
            centralized=average_metrics(centralized),
            single_training=average_metrics(single),
        )
        sim_metrics.append(sm)

        gen_metrics_df = pd.read_csv(exp_path / str(j) / "generalization_metrics.csv")
        gossip = [
            Metrics(
                rmse=gen_metrics_df.iloc[i + 3]["rmse"],
                mae=gen_metrics_df.iloc[i + 3]["mae"],
                mape=gen_metrics_df.iloc[i + 3]["mape"],
                msle=gen_metrics_df.iloc[i + 3]["msle"],
                mse=gen_metrics_df.iloc[i + 3]["mse"],
            )
            for i in range(n_nodes)
        ]
        single = [
            Metrics(
                rmse=gen_metrics_df.iloc[i + n_nodes + 3]["rmse"],
                mae=gen_metrics_df.iloc[i + n_nodes + 3]["mae"],
                mape=gen_metrics_df.iloc[i + n_nodes + 3]["mape"],
                msle=gen_metrics_df.iloc[i + n_nodes + 3]["msle"],
                mse=gen_metrics_df.iloc[i + n_nodes + 3]["mse"],
            )
            for i in range(n_nodes)
        ]
        centralized = [
            Metrics(
                rmse=gen_metrics_df.iloc[2]["rmse"],
                mae=gen_metrics_df.iloc[2]["mae"],
                mape=gen_metrics_df.iloc[2]["mape"],
                msle=gen_metrics_df.iloc[2]["msle"],
                mse=gen_metrics_df.iloc[2]["mse"],
            )
        ]
        sm = SimulationMetrics(
            gossip=average_metrics(gossip),
            centralized=average_metrics(centralized),
            single_training=average_metrics(single),
        )
        gen_metrics.append(sm)

    return sim_metrics, gen_metrics
