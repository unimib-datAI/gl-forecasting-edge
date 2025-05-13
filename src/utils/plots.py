import numpy as np
from matplotlib import pyplot as plt

from utils.metrics import Metrics


def comparison_plot(experiments: list[tuple[str, Metrics]], title: str, file: str) -> None:
    width = 0.8
    x_ticks = [name for (name, metrics) in experiments]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    axes[0, 0].bar(
        x_ticks,
        np.array([
            exp_metrics.rmse for name, exp_metrics in experiments
        ]),
        width=width,
    )
    axes[0, 0].set_title("Root Mean Squared Error")

    axes[0, 1].bar(
        x_ticks,
        np.array([
            exp_metrics.mse for name, exp_metrics in experiments
        ]),
        width=width,
    )
    axes[0, 1].set_title("Mean Squared Error")

    axes[1, 0].bar(
        x_ticks,
        np.array([
            exp_metrics.msle for name, exp_metrics in experiments
        ]),
        width=width,
    )
    axes[1, 0].set_title("Mean Squared Log Error")

    axes[1, 1].bar(
        x_ticks,
        np.array([
            exp_metrics.mae for name, exp_metrics in experiments
        ]),
        width=width,
    )
    axes[1, 1].set_title("Mean Absolute Error")

    fig.suptitle(title)

    plt.savefig(file, dpi=500)