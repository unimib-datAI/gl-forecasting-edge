import json
import multiprocessing as mp
from enum import StrEnum
from pathlib import Path
from typing import Optional, Sequence

import networkx as nx
import numpy as np
from keras.api.saving import load_model
from matplotlib import pyplot as plt

from gossiplearning import History
from gossiplearning.config import Config
from gossiplearning.history import NodeMetricsHistory
from gossiplearning.models import LabelledData, NodeId
from utils.data import get_test_sets
from utils.metrics import (
    SimulationMetrics,
    plot_metrics_violinplot,
    plot_predicted_time_series,
    dump_metrics,
    average_metrics,
    compute_metrics,
    compute_generalized_metrics,
    Metrics,
    SimulationNodeMetrics,
)


class EvaluationMode(StrEnum):
    SINGLE = "single"
    GOSSIP = "gossip"
    CENTRALIZED = "centralized"


def evaluate_simulations(
    n_sim: int,
    config: Config,
    dataset_base_dir: Path,
    evaluate_generalization: bool,
    ds_name: str,
    eval_drop_tower: bool,
    network_dir: Path,
    start: int = 0,
) -> list[tuple[SimulationNodeMetrics, Optional[SimulationMetrics]]]:
    """Evaluate multiple simulation and save plots exploiting multiprocessing."""
    with mp.Pool(3) as p:
        results = p.starmap(
            evaluate_simulation,
            [
                (
                    i,
                    config,
                    dataset_base_dir,
                    evaluate_generalization,
                    ds_name,
                    network_dir / str(i),
                    eval_drop_tower,
                )
                for i in range(start, start + n_sim)
            ],
        )
        return zip(*results)


def evaluate_simulation(
    sim_number: int,
    config: Config,
    datasets_base_dir: Path,
    evaluate_generalization: bool,
    ds_name: str,
    network_dir: Path,
    eval_drop_tower: bool,
) -> tuple[SimulationNodeMetrics, Optional[SimulationMetrics]]:
    """Evaluate a simulation and save its plots."""
    simulation_workspace = Path(config.workspace_dir) / str(sim_number)
    plots_dir = simulation_workspace / "plots"
    gossip_models_dir = simulation_workspace / "models"
    datasets_dir = datasets_base_dir / str(sim_number) / ds_name
    network = nx.read_adjlist(network_dir / "adj_list.txt", nodetype=int)

    try:
        with (datasets_dir / "scaling_factor.txt").open("r") as f:
            scaling_factor = float(f.readline())
    except Exception as e:
        print("Scaling factor not defined!")
        scaling_factor = 1

    test_sets = get_test_sets(datasets_dir, config.n_nodes)

    predictions_folder = simulation_workspace / "predictions"
    predictions_folder.mkdir(parents=True, exist_ok=True)

    if (predictions_folder / "gossip_0.npz").exists():
        gossip_pred = {}
        single_pred = {}
        for i in range(config.n_nodes):
            pred = np.load(predictions_folder / f"gossip_{i}.npz")
            gossip_pred[i] = {int(j): p for j, p in pred.items()}
            pred = np.load(predictions_folder / f"single_{i}.npz")
            single_pred[i] = {int(j): p for j, p in pred.items()}

        pred = np.load(predictions_folder / "centralized.npz")
        centralized_pred = {int(j): p for j, p in pred.items()}
    else:
        gossip_pred, single_pred, centralized_pred = _compute_predictions(
            n_nodes=config.n_nodes,
            datasets_dir=datasets_dir,
            gossip_models_dir=gossip_models_dir,
            test_sets=test_sets,
            network=network,
            scaling_factor=scaling_factor,
        )

        for i in range(config.n_nodes):
            np.savez(
                predictions_folder / f"gossip_{i}.npz",
                **{str(j): pred for j, pred in gossip_pred[i].items()},
            )
            np.savez(
                predictions_folder / f"single_{i}.npz",
                **{str(j): pred for j, pred in single_pred[i].items()},
            )
        np.savez(
            predictions_folder / "centralized.npz",
            **{str(j): pred for j, pred in centralized_pred.items()},
        )

    # after prediction, let's scale back the truth
    truths = [ts[1] * scaling_factor for ts in test_sets]

    gossip_metrics, single_metrics, centralized_metrics = _evalute_predictions(
        gossip_pred=gossip_pred,
        single_pred=single_pred,
        centralized_pred=centralized_pred,
        n_nodes=config.n_nodes,
        truth=[truth for truth in truths],
    )

    _plot_and_dump_sim_metrics(
        gossip_metrics=gossip_metrics,
        single_metrics=single_metrics,
        centralized_metrics=centralized_metrics,
        simulation_workspace=simulation_workspace,
        plots_dir=plots_dir,
        n_nodes=config.n_nodes,
        n_output_vars=config.training.n_output_vars,
    )

    for j in range(config.n_nodes):
        file = plots_dir / "predictions" / f"node_{j}_predictions.svg"
        plot_predicted_time_series(
            gossip_predictions=gossip_pred[j][j],
            single_predictions=single_pred[j][j],
            centralized_predictions=centralized_pred[j],
            truth=truths[j],
            file=file,
            n_functions=config.training.n_output_vars,
            end_at=500,
        )

    gossip_gen_metrics, single_gen_metrics, centralized_gen_metrics = (
        _compute_and_dump_generalization_metrics(
            test_sets=test_sets,
            gossip_pred=gossip_pred,
            single_pred=single_pred,
            centralized_pred=centralized_pred,
            simulation_workspace=simulation_workspace,
            plots_dir=plots_dir,
            n_nodes=config.n_nodes,
            network=network,
        )
        if evaluate_generalization
        else ([], [], [])
    )

    drop_antenna_metrics = None
    if eval_drop_tower:
        drop_antenna_metrics = evaluate_drop_antenna(
            n_nodes=config.n_nodes,
            workspace_dir=simulation_workspace,
            network=network,
            test_sets=test_sets,
            datasets_dir=datasets_dir,
        )

    sim_node_metrics = SimulationNodeMetrics(
        gossip=gossip_metrics,
        single=single_metrics,
        centralized=centralized_metrics,
        gossip_generalized=gossip_gen_metrics,
        single_generalized=single_gen_metrics,
        centralized_generalized=centralized_gen_metrics,
    )

    return sim_node_metrics, drop_antenna_metrics


def _get_node_predictions(
    n_nodes: int,
    models_dir: Path,
    test_sets: Sequence[LabelledData],
    network: nx.Graph,
    scaling_factor: float = 1,
    single: bool = False,
) -> dict[NodeId, dict[NodeId, np.ndarray]]:
    result: dict[NodeId, dict[NodeId, np.ndarray]] = {}
    for i in range(n_nodes):
        model_name = f"{i}_single.keras" if single else f"{i}.keras"
        model = load_model(models_dir / model_name)

        predict_test_sets = [i, *network.neighbors(i)]

        node_predictions = {
            idx: model.predict(test_sets[idx][0], verbose=0) * scaling_factor
            for idx in predict_test_sets
        }

        result[i] = node_predictions

    return result


def _get_centralized_predictions(
    model_path: Path,
    test_sets: Sequence[LabelledData],
    scaling_factor: float = 1,
) -> dict[NodeId, np.ndarray]:
    model = load_model(model_path)

    return {
        node_id: model.predict(test_set[0], verbose=0) * scaling_factor
        for node_id, test_set in enumerate(test_sets)
    }


def evaluate_drop_antenna(
    n_nodes: int,
    workspace_dir: Path,
    network: nx.Graph,
    test_sets: Sequence[LabelledData],
    datasets_dir: Path,
) -> SimulationMetrics:
    gossip_pred = _get_predictions_with_drop_antenna(
        n_nodes=n_nodes,
        models_dir=workspace_dir / "models",
        test_sets=test_sets,
        network=network,
        eval_mode=EvaluationMode.GOSSIP,
    )

    single_pred = _get_predictions_with_drop_antenna(
        n_nodes=n_nodes,
        models_dir=datasets_dir / "models",
        test_sets=test_sets,
        network=network,
        eval_mode=EvaluationMode.SINGLE,
    )

    centralized_pred = _get_predictions_with_drop_antenna(
        n_nodes=n_nodes,
        models_dir=datasets_dir / "models",
        test_sets=test_sets,
        network=network,
        eval_mode=EvaluationMode.CENTRALIZED,
    )

    single_metrics, gossip_metrics, centralized_metrics = _drop_antenna_metrics(
        single=single_pred, gossip=gossip_pred, centralized=centralized_pred
    )

    simulation_metrics = SimulationMetrics(
        gossip=average_metrics(gossip_metrics),
        single_training=average_metrics(single_metrics),
        centralized=average_metrics(centralized_metrics),
    )

    metrics_out_file = workspace_dir / "metrics_drop_antenna.csv"
    dump_metrics(
        gossip_metrics,
        single_metrics,
        centralized_metrics,
        simulation_metrics,
        metrics_out_file,
    )

    plots_dir = workspace_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    metrics_dir = plots_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)

    metrics_boxplot_file = metrics_dir / "metrics_boxplot_drop_antenna.svg"
    plot_metrics_violinplot(
        # [
        #     SimulationMetrics(
        #         gossip=gossip_metrics[i],
        #         single_training=single_metrics[i],
        #         centralized=centralized_metrics[i],
        #     )
        #     for i in range(n_nodes)
        # ],
        gossip=gossip_metrics,
        single=single_metrics,
        centralized=centralized_metrics,
        file=metrics_boxplot_file,
    )

    return simulation_metrics


def _get_predictions_with_drop_antenna(
    n_nodes: int,
    models_dir: Path,
    test_sets: Sequence[LabelledData],
    network: nx.Graph,
    eval_mode: EvaluationMode,
) -> Sequence[tuple[np.ndarray, np.ndarray]]:
    result: list[tuple[np.ndarray, np.ndarray]] = []

    for node in range(n_nodes):
        match eval_mode:
            case EvaluationMode.SINGLE:
                model_name = f"{node}_single.keras"
            case EvaluationMode.GOSSIP:
                model_name = f"{node}.keras"
            case EvaluationMode.CENTRALIZED:
                model_name = "centralized.keras"
            case _:
                raise Exception("unknown eval mode")

        model = load_model(models_dir / model_name)

        node_result = []

        neighbors = network.neighbors(node)
        for neighbor in neighbors:
            neighbor_degree = network.degree[neighbor]
            X = test_sets[node][0] + test_sets[neighbor][0] / neighbor_degree
            Y = test_sets[node][1] + test_sets[neighbor][1] / neighbor_degree

            pred = model.predict(X, verbose=0)
            node_result.append((pred, Y))

        pred = np.concatenate([pred for (pred, _) in node_result])
        truth = np.concatenate([Y for (_, Y) in node_result])

        result.append((pred, truth))
    return result


def _drop_antenna_metrics(
    single: Sequence[tuple[np.ndarray, np.ndarray]],
    gossip: Sequence[tuple[np.ndarray, np.ndarray]],
    centralized: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[Sequence[Metrics], Sequence[Metrics], Sequence[Metrics]]:
    single_metrics = [compute_metrics(truth, predicted) for predicted, truth in single]
    gossip_metrics = [compute_metrics(truth, predicted) for predicted, truth in gossip]
    centralized_metrics = [
        compute_metrics(truth, predicted) for predicted, truth in centralized
    ]

    return single_metrics, gossip_metrics, centralized_metrics


def plot_node_history(
    history: NodeMetricsHistory,
    file: Path,
) -> None:
    plt.ioff()
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(
        history["loss"],
        label="Training",
        color="royalblue",
        linewidth=1,
    )

    ax.plot(
        history["val_loss"],
        label="Validation",
        color="tomato",
        linewidth=1,
    )

    ax.legend()
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")

    plt.savefig(file, format="svg", dpi=500)
    plt.close()


def _compute_predictions(
    n_nodes: int,
    gossip_models_dir: Path,
    test_sets: Sequence[LabelledData],
    datasets_dir: Path,
    network: nx.Graph,
    scaling_factor: float,
) -> tuple[
    dict[NodeId, dict[NodeId, np.ndarray]],
    dict[NodeId, dict[NodeId, np.ndarray]],
    dict[NodeId, np.ndarray],
]:
    gossip_pred = _get_node_predictions(
        n_nodes=n_nodes,
        models_dir=gossip_models_dir,
        test_sets=test_sets,
        single=False,
        network=network,
        scaling_factor=scaling_factor,
    )

    single_pred = _get_node_predictions(
        n_nodes=n_nodes,
        models_dir=datasets_dir / "models",
        test_sets=test_sets,
        single=True,
        network=network,
        scaling_factor=scaling_factor,
    )

    centralized_pred = _get_centralized_predictions(
        model_path=datasets_dir / "models" / "centralized.keras",
        test_sets=test_sets,
        scaling_factor=scaling_factor,
    )

    return gossip_pred, single_pred, centralized_pred


def _evalute_predictions(
    gossip_pred: dict[NodeId, dict[NodeId, np.ndarray]],
    single_pred: dict[NodeId, dict[NodeId, np.ndarray]],
    centralized_pred: dict[NodeId, np.ndarray],
    truth: Sequence[np.ndarray],
    n_nodes: int,
) -> tuple[Sequence[Metrics], Sequence[Metrics], Sequence[Metrics]]:
    gossip_metrics = [
        compute_metrics(truth[i], gossip_pred[i][i]) for i in range(n_nodes)
    ]
    single_metrics = [
        compute_metrics(truth[i], single_pred[i][i]) for i in range(n_nodes)
    ]
    centralized_metrics = [
        compute_metrics(truth[i], centralized_pred[i]) for i in range(n_nodes)
    ]

    return gossip_metrics, single_metrics, centralized_metrics


def _plot_and_dump_sim_metrics(
    gossip_metrics: Sequence[Metrics],
    single_metrics: Sequence[Metrics],
    centralized_metrics: Sequence[Metrics],
    simulation_workspace: Path,
    plots_dir: Path,
    n_nodes: int,
    n_output_vars: int,
) -> None:
    simulation_metrics = SimulationMetrics(
        gossip=average_metrics(gossip_metrics),
        single_training=average_metrics(single_metrics),
        centralized=average_metrics(centralized_metrics),
    )

    metrics_out_file = simulation_workspace / "metrics.csv"
    dump_metrics(
        gossip_metrics,
        single_metrics,
        centralized_metrics,
        simulation_metrics,
        metrics_out_file,
    )

    metrics_dir = plots_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True, parents=True)
    plot_metrics_violinplot(
        gossip=gossip_metrics,
        single=single_metrics,
        centralized=centralized_metrics,
        folder=metrics_dir,
    )

    (plots_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (plots_dir / "training").mkdir(parents=True, exist_ok=True)

    history_json = json.load((simulation_workspace / "history.json").open("r"))
    history = History(**history_json)

    for j, node_training_history in enumerate(history.nodes_training_history.values()):
        for fn in range(n_output_vars):
            file = plots_dir / "training" / f"node_{j}_gossip_fn_{fn}.svg"
            plot_node_history(
                history=node_training_history,
                file=file,
            )


def _compute_and_dump_generalization_metrics(
    test_sets: Sequence[LabelledData],
    gossip_pred: Sequence[Sequence[np.ndarray]],
    single_pred: Sequence[Sequence[np.ndarray]],
    centralized_pred: Sequence[np.ndarray],
    simulation_workspace: Path,
    plots_dir: Path,
    n_nodes: int,
    network: nx.Graph,
) -> tuple[list[Metrics], list[Metrics], list[Metrics]]:
    gossip_generalization_metrics = []
    single_generalization_metrics = []
    centralized_generalization_metrics = []

    for i in range(n_nodes):
        neighbors = [*network.neighbors(i)]
        joined_test_labels = tuple(test_sets[node][1] for node in neighbors)

        gossip_generalization_metrics.append(
            compute_generalized_metrics(
                joined_test_labels, [gossip_pred[i][node] for node in neighbors]
            )
        )

        single_generalization_metrics.append(
            compute_generalized_metrics(
                joined_test_labels, [single_pred[i][node] for node in neighbors]
            )
        )

        centralized_generalization_metrics.append(
            compute_generalized_metrics(
                joined_test_labels, [centralized_pred[node] for node in neighbors]
            )
        )

    simulation_generalization_metrics = SimulationMetrics(
        gossip=average_metrics(gossip_generalization_metrics),
        single_training=average_metrics(single_generalization_metrics),
        centralized=average_metrics(centralized_generalization_metrics),
    )

    gen_metrics_out_file = simulation_workspace / "generalization_metrics.csv"
    dump_metrics(
        gossip_generalization_metrics,
        single_generalization_metrics,
        centralized_generalization_metrics,
        simulation_generalization_metrics,
        gen_metrics_out_file,
    )

    metrics_folder = plots_dir / "metrics"

    plot_metrics_violinplot(
        gossip=gossip_generalization_metrics,
        single=single_generalization_metrics,
        centralized=centralized_generalization_metrics,
        folder=metrics_folder,
        file_prefix="generalization",
    )

    return (
        gossip_generalization_metrics,
        single_generalization_metrics,
        centralized_generalization_metrics,
    )
