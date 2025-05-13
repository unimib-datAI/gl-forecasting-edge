from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from gossiplearning.history import History


def plot_history(
    history: History, *, file: str, only_nodes: Optional[tuple[int, ...]]
) -> None:
    """
    Plot the gossip protocol history.

    :param history: the history object containing all events
    :param file: the file where the plot should be saved
    :param only_nodes: highlight events only for these nodes, if provided
    """
    n_nodes = len(history.stopped_time)

    plt.ioff()  # in this way plots are just saved and not shown

    fig, ax = plt.subplots()

    node_indices = np.arange(0, n_nodes)
    plt.yticks(ticks=node_indices, labels=[f"Node {i}" for i in node_indices])

    for node, time in history.stopped_time.items():
        ax.plot(
            (0, time), (node, node), color="grey", linestyle="dashed", linewidth=0.5
        )
        ax.plot(time, node, color="red", marker="x")

    for msg in history.messages:
        if (
            only_nodes is not None
            and msg.from_node not in only_nodes
            and msg.to_node not in only_nodes
        ):
            continue

        stopped_sender = history.stopped_time[msg.from_node] < msg.time_sent
        stopped_receiver = history.stopped_time[msg.to_node] < msg.time_received

        ax.plot(
            (msg.time_sent, msg.time_received),
            (msg.from_node, msg.to_node),
            color="blue" if not stopped_sender else "grey",
            linestyle="dashed",
            linewidth=0.3 if not stopped_sender else 0.15,
        )

        ax.plot(msg.time_sent, msg.from_node, color="blue", marker="^", markersize=1)
        ax.plot(
            msg.time_received,
            msg.to_node,
            color="blue" if not stopped_receiver else "red",
            marker="v" if not stopped_receiver else "x",
            markersize=1 if not stopped_receiver else 2,
        )

    for train_slot in history.trainings:
        if only_nodes is None or train_slot.node in only_nodes:
            ax.plot(
                (train_slot.from_time, train_slot.to_time),
                (train_slot.node, train_slot.node),
                color="blue",
                linewidth=0.5,
            )

    plt.savefig(file, dpi=500)
    plt.close(fig)
