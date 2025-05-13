from dataclasses import dataclass
from functools import singledispatch

import numpy as np

from gossiplearning import History
from gossiplearning.config import Config
from gossiplearning.history import FailureHistoryLog, MessageHistoryLog, UpdateHistoryLog
from gossiplearning.log import Logger
from gossiplearning.models import NodeId, ModelWeights, Loss, Time, WeightsMessage
from gossiplearning.node import Node, NodeState


@dataclass
class Event:
    time: int
    handler_node_id: NodeId | None

    def __lt__(self, other: "Event"):
        return self.time < other.time

@dataclass
class IsTimeToFailEvent(Event):
    pass

@dataclass
class FailedNodeEvent(Event):
    pass

@dataclass
class RecoveryNodeEvent(Event):
    pass

@dataclass
class SendModelsLoopEvent(Event):
    pass


@dataclass
class ReceiveModelEvent(Event):
    received_msg: WeightsMessage
    from_node_id: NodeId
    sent_at: Time


@dataclass
class SaveModelEvent(Event):
    latest_weights: ModelWeights
    best_update_model_weights: ModelWeights
    best_update_val_loss: Loss
    updates_without_improving: int
    new_weight: int
    trained_started_at: Time


TRAIN_TIME: int = 5


@singledispatch
def process_event(
    event: Event,
    node: Node,
    logger: Logger,
    history: History,
    config: Config,
    nodes: list[Node] | None = None,
) -> tuple[Event, ...]:
    raise Exception("Event type not recognized!")

@process_event.register
def process_is_time_to_fail_event(
    event: IsTimeToFailEvent,
    node: None,
    logger: Logger,
    history: History,
    config: Config,
    nodes: list[Node]
) -> tuple[Event, ...]:
    result: list[Event] = []
    node_ids = np.random.choice(range(config.n_nodes), size=int(config.n_nodes * config.training.max_percentage_failed_nodes), replace=False)
    logger.debug_log(f"Nodes {node_ids} selected for potential failure")

    for node_id in node_ids:
        has_to_fail = np.random.choice([True, False], p=[config.training.node_failure_probability, 1 - config.training.node_failure_probability])
        logger.debug_log(f"Node {node_id} has to fail: {has_to_fail}")

        if has_to_fail:
            failure_time = event.time + 1
            recovery_time = failure_time + int(np.random.normal(config.training.node_recovery_time_mean, config.training.node_recovery_time_std, size=1))

            if nodes[node_id].state == NodeState.FAILED:
                logger.debug_log(f"Node {node_id} is already failed! Skipping...")
                continue

            logger.debug_log(f"Node {node_id} will fail at {failure_time} and recover at {recovery_time}")
            result.append(FailedNodeEvent(time=failure_time, handler_node_id=node_id))
            result.append(RecoveryNodeEvent(time=recovery_time, handler_node_id=node_id))
    
    next_failure_check_time = event.time + config.training.is_time_to_fail_frequency
    result.append(IsTimeToFailEvent(time=next_failure_check_time, handler_node_id=None))
    logger.debug_log(f"Next failure check at {next_failure_check_time}")

    return tuple(result)

@process_event.register
def process_failed_node_event(
    event: FailedNodeEvent,
    node: Node,
    logger: Logger,
    history: History,
    config: Config,
    nodes: list[Node] | None = None,
) -> tuple[Event, ...]:
    node.state_before_failure = node.state
    node.state = NodeState.FAILED
    if node.id not in history.nodes_failures_history.keys():
        history.nodes_failures_history[node.id] = []
    history.nodes_failures_history[node.id].append(FailureHistoryLog(failed_at=event.time, recovered_at=None))
    logger.node_event_log(
        "Node failed", time=event.time, node=event.handler_node_id
    )
    return ()

@process_event.register
def process_recovery_node_event(
    event: RecoveryNodeEvent,
    node: Node,
    logger: Logger,
    history: History,
    config: Config,
    nodes: list[Node] | None = None,
) -> tuple[Event, ...]:
        
    node.state = node.state_before_failure if node.state_before_failure is not NodeState.TRAINING else NodeState.ACTIVE
    history.nodes_failures_history[node.id][-1].recovered_at = event.time
    logger.node_event_log(
        "Node recovered from failure", time=event.time, node=event.handler_node_id
    )

    results = []
    results.append(SendModelsLoopEvent(time=event.time + 1, handler_node_id=node.id))
    return tuple(results)

@process_event.register
def process_send_model_event(
    event: SendModelsLoopEvent,
    node: Node,
    logger: Logger,
    history: History,
    config: Config,
    nodes: list[Node] | None = None,
) -> tuple[Event, ...]:
    
    if node.state == NodeState.FAILED:
        logger.node_event_log(f"Cannot send model weights due to node failure", time=event.time, node=event.handler_node_id)
        return ()

    logger.node_event_log(
        f"Sending models", time=event.time, node=event.handler_node_id
    )

    # negligible time in checking this, not simulated
    node.active_links = [
        link for link in node.active_links if link.node not in history.stopped_time
    ]

    if len(node.active_links) == 0:
        logger.debug_log(f"Node {node.id} has no active neighbors! Stop sending model weights")
        return ()

    n_selected = np.ceil(
        len(node.active_links) * config.training.target_probability
    ).astype(int)

    selected_indices = np.random.choice(
        np.arange(len(node.active_links)),
        replace=False,
        size=n_selected,
    )

    selected_links = [node.active_links[ind] for ind in selected_indices]

    logger.debug_log(
        f"Node {node.id} will send model weights to the following nodes: {[link.node for link in selected_links]}"
    )

    result: list[Event] = []

    # for each target node, create a "receive model" event

    message = node.marshal_model()

    for target in selected_links:
        arrival_time = event.time + target.weights_transmission_time

        history.messages.append(
            MessageHistoryLog(
                from_node=node.id,
                to_node=target.node,
                time_sent=event.time,
                time_received=arrival_time,
            )
        )

        result.append(
            ReceiveModelEvent(
                time=arrival_time,
                handler_node_id=target.node,
                received_msg=message,
                from_node_id=node.id,
                sent_at=event.time,
            )
        )

    # for this node, create the next "send model" event
    next_send_time = event.time + max(
        [link.weights_transmission_time for link in selected_links]
    )

    result.append(SendModelsLoopEvent(time=next_send_time, handler_node_id=node.id))
    return tuple(result)


@process_event.register
def process_receive_model_event(
    event: ReceiveModelEvent,
    node: Node,
    logger: Logger,
    history: History,
    config: Config,
    nodes: list[Node] | None = None,
) -> tuple[Event, ...]:
    
    if node.state == NodeState.FAILED:
        logger.node_event_log(f"Receiving model weights from node {event.from_node_id} failed due to node failure", time=event.time, node=event.handler_node_id)
        return ()

    if event.handler_node_id in history.stopped_time:
        return ()

    logger.node_event_log(
        "Receiving model", time=event.time, node=event.handler_node_id
    )

    if node.state == NodeState.TRAINING:
        logger.debug_log(
            f"Node {node.id} is already training! Received model weights are discarded"
        )
        return ()

    logger.debug_log(f"Node {node.id} merged model weights! Training started...")

    node.receive_weights(event.received_msg, event.from_node_id)

    result: list[Event] = []

    if node.ready_to_train:
        node.merge_models()

        (
            latest_weights,
            best_update_weights,
            val_loss,
            updates_without_improving,
        ) = node.perform_update()

        # TODO: change static train time or improved estimation
        finish_train_time = event.time + TRAIN_TIME

        result.append(
            SaveModelEvent(
                time=finish_train_time,
                handler_node_id=node.id,
                latest_weights=latest_weights,
                best_update_model_weights=best_update_weights,
                best_update_val_loss=val_loss,
                updates_without_improving=updates_without_improving,
                new_weight=node.accumulated_weight + node.weight,
                trained_started_at=event.time,
            )
        )

        history.trainings.append(
            UpdateHistoryLog(
                node=node.id, from_time=event.time, to_time=finish_train_time
            )
        )

    return tuple(result)


@process_event.register
def process_save_model_event(
    event: SaveModelEvent,
    node: Node,
    logger: Logger,
    history: History,
    config: Config,
    nodes: list[Node] | None = None,
) -> tuple[Event, ...]:
    
    if node.state == NodeState.FAILED:
        logger.node_event_log(f"Cannot save model due to node failure", time=event.time, node=event.handler_node_id)
        return ()

    if event.handler_node_id in history.stopped_time:
        return ()

    logger.node_event_log(
        "Completed one update! Saving model",
        node=event.handler_node_id,
        time=event.time,
    )

    node.save_model(
        latest_weights=event.latest_weights,
        best_update_model_weights=event.best_update_model_weights,
        time=event.time,
        best_update_val_loss=event.best_update_val_loss,
        updates_without_improving=event.updates_without_improving,
        new_model_weight=event.new_weight,
    )

    # if (
    #     node.state == NodeState.STOPPED or node.state_before_failure == NodeState.STOPPED
    #     and event.handler_node_id not in history.stopped_time.keys()
    # ):
    #     logger.debug_log(f"Node {event.handler_node_id} becomes inactive")

    #     history.stopped_time[event.handler_node_id] = event.time

    if (
        node.state == NodeState.STOPPED and event.handler_node_id not in history.stopped_time.keys()
    ):
        logger.debug_log(f"Node {event.handler_node_id} becomes inactive")

        history.stopped_time[event.handler_node_id] = event.time

    return ()
