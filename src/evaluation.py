import json
import os
from pathlib import Path

from gossiplearning.config import Config
from utils.evaluation import evaluate_simulations
from utils.metrics import (
    SimulationMetrics,
    average_metrics,
    plot_metrics_violinplot,
    dump_experiment_metrics,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

dataset_base_dir = Path(f"data/datasets/porto_10n_3k")
network_dir = Path(f"data/networks/porto_10n_3k")
ds_name = "4in_notscaled"

SIMULATIONS_PER_CONFIG = 10

evaluate_generalization = True
evaluate_drop_antenna = False


if __name__ == "__main__":
    for i in range(82, 83):
        exp = f"experiments/porto_{i}"
        with open(f"{exp}/0/config.json", "r") as f:
            config = json.load(f)

        config = Config.model_validate(config)
        config.workspace_dir = Path(exp)

        aggregated_plots = Path(config.workspace_dir) / "plots"
        aggregated_plots.mkdir(exist_ok=True, parents=True)

        (
            sim_node_metrics,
            drop_antenna_metrics,
        ) = evaluate_simulations(
            n_sim=SIMULATIONS_PER_CONFIG,
            config=config,
            dataset_base_dir=dataset_base_dir,
            evaluate_generalization=evaluate_generalization,
            eval_drop_tower=False,
            start=0,
            ds_name=ds_name,
            network_dir=network_dir,
        )

        gossip_metrics = [m for sm in sim_node_metrics for m in sm.gossip]
        single_metrics = [m for sm in sim_node_metrics for m in sm.single]
        centralized_metrics = [m for sm in sim_node_metrics for m in sm.centralized]
        gossip_gen_metrics = [
            m for sm in sim_node_metrics for m in sm.gossip_generalized
        ]
        single_gen_metrics = [
            m for sm in sim_node_metrics for m in sm.single_generalized
        ]
        centralized_gen_metrics = [
            m for sm in sim_node_metrics for m in sm.centralized_generalized
        ]

        averaged_metrics = SimulationMetrics(
            gossip=average_metrics(gossip_metrics),
            single_training=average_metrics(single_metrics),
            centralized=average_metrics(centralized_metrics),
        )

        plot_metrics_violinplot(
            gossip=gossip_metrics,
            single=single_metrics,
            centralized=centralized_metrics,
            folder=aggregated_plots,
        )

        dump_experiment_metrics(
            gossip_metrics=gossip_metrics,
            single_metrics=single_metrics,
            centralized=centralized_metrics,
            file=Path(config.workspace_dir) / "metrics.csv",
        )

        if evaluate_generalization:
            averaged_generalization_metrics = SimulationMetrics(
                gossip=average_metrics(gossip_gen_metrics),
                single_training=average_metrics(single_gen_metrics),
                centralized=average_metrics(centralized_gen_metrics),
            )

            plot_metrics_violinplot(
                gossip=gossip_gen_metrics,
                single=single_gen_metrics,
                centralized=centralized_gen_metrics,
                folder=aggregated_plots,
                file_prefix="generalization",
            )

            dump_experiment_metrics(
                gossip_metrics=gossip_gen_metrics,
                single_metrics=single_gen_metrics,
                centralized=centralized_gen_metrics,
                file=Path(config.workspace_dir) / "metrics_generalized.csv",
            )
