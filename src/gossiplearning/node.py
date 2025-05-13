import math
from enum import IntEnum
from pathlib import Path
from typing import Optional

from gossiplearning.config import TrainingConfig, HistoryConfig
from gossiplearning.log import Logger
from gossiplearning.models import (
    StopCriterion,
    ModelWeights,
    Loss,
    NodeId,
    MetricValue,
    MetricName,
    ModelBuilder,
    NodeDataFn,
    Dataset,
    Link,
    AggregatorFn,
    WeightsMessage,
    LabelledData,
    NodeWeightFn,
)
from gossiplearning.weights_marshaling import MarshalWeightsFn
from utils.metrics import compute_metrics, Metrics


class NodeState(IntEnum):
    ACTIVE = 0
    STOPPED = 1
    TRAINING = 2
    FAILED = 3


class Node:
    """
    A gossip learning node.
    """

    def __init__(
        self,
        *,
        create_model_fn: ModelBuilder,
        id: NodeId,
        links: tuple[Link, ...],
        training_config: TrainingConfig,
        history_config: HistoryConfig,
        workspace_dir: Path,
        logger: Logger,
        node_data_fn: NodeDataFn,
        aggregator: AggregatorFn,
        marshal_weights_fn: MarshalWeightsFn,
        test_set: LabelledData,
        weight_fn: NodeWeightFn,
    ) -> None:
        """
        Initialize the node for gossip protocol.

        :param create_model_fn: the function used for creating a model
        :param id: the node identifier
        :param links: the set of node links
        :param training_config: the global training & gossip configuration
        :param workspace_dir: the workspace base directory
        """
        # internal state

        self._model = create_model_fn()
        self._create_model = create_model_fn
        self._logger = logger

        self._training_config = training_config
        self._history_config = history_config
        self._workspace_dir = workspace_dir

        self.data: Dataset = node_data_fn(id)

        self._last_improved_time = 0
        self._updates_without_improving = 0
        self._best_val_loss: Loss = math.inf
        self._best_weights: Optional[ModelWeights] = None
        self._completed_updates = 0

        self._received_weights: dict[NodeId, WeightsMessage] = {}
        self._aggregator = aggregator
        self._marshal_weights_fn = marshal_weights_fn
        self._test_set = test_set

        # public state
        self.id = id
        self.accumulated_weight = 0
        self.active_links = list(links)
        self.training_history: dict[MetricName, list[MetricValue]] = {}
        self.state = NodeState.ACTIVE
        self.state_before_failure = NodeState.ACTIVE
        self.n_training_samples = len(self.data["X_train"])
        self.eval_metrics: list[Metrics] = []
        self.weight = weight_fn(self.data)

    def merge_models(self) -> None:
        """
        Merge all the received model weights into the current model.

        The internal model weights are updated. The number of trained samples is set at the
        maximum between the number of trained samples of the merged models.
        """
        self._model, self.accumulated_weight = self._aggregator(
            self._model,
            self.accumulated_weight,
            tuple(msg for k, msg in self._received_weights.items()),
        )

        self._received_weights = {}

    def perform_update(self) -> tuple[ModelWeights, ModelWeights, Loss, int]:
        """
        Perform a model update, training the node model on local data for a given number of epochs.

        The number of epochs is the one specific in the global training configuration object.

        Set the node state to TRAINING and leave it on, in order to stop reception of new models
        until the current one will be saved.

        :return: latest model weights, weights of the best trained model, its loss and the current number of updates without improvements
        """
        self.state = NodeState.TRAINING

        latest_weights, best_weights, best_val_loss = self.train_model(
            n_epochs=self._training_config.epochs_per_update,
        )

        self._completed_updates += 1

        self._evaluate()

        return (
            latest_weights,
            best_weights,
            best_val_loss,
            self._updates_without_improving,
        )

    def train_model(self, n_epochs: int) -> tuple[ModelWeights, ModelWeights, Loss]:
        """
        Train the node model on local data for the specified number of epochs.

        After every epoch, store the training and validation metrics that will be used to build the
        training history. Also, keep track of the best obtained validation loss (among the performed
        epochs) and the related weights and return them.

        :param n_epochs: the number of training epochs.
        :return: latest model weights, weights of the best trained model and best validation loss
        """
        if n_epochs < 1:
            raise Exception("Epochs number must be at least 1!")

        model = self._create_model()
        model.set_weights(self._model.get_weights())

        best_val_loss = math.inf
        best_weights = None
        for i in range(n_epochs):
            history = model.fit(
                self.data["X_train"],
                self.data["Y_train"],
                epochs=1,
                validation_data=(self.data["X_val"], self.data["Y_val"]),
                verbose=0,
                batch_size=self._training_config.batch_size,
                validation_batch_size=self._training_config.batch_size,
                shuffle=self._training_config.shuffle_batch,
            ).history

            if len(model.loss) > 1:
                val_loss = sum([history[f"val_{l}_loss"][0] for l in model.loss.keys()])
            else:
                val_loss = history["val_loss"][0]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights()

            for metric in history:
                metric_value = history[metric][0]

                if metric not in self.training_history:
                    self.training_history[metric] = [metric_value]
                else:
                    self.training_history[metric].append(metric_value)

        assert best_weights
        latest_weights = model.get_weights()
        return latest_weights, best_weights, best_val_loss

    def marshal_model(self) -> WeightsMessage:
        """
        Sample weights from the current model accordingly to the percentage specified in the config.

        :return: the sampled weights
        """
        return WeightsMessage(
            marshaled_weights=self._marshal_weights_fn(
                self._model, self._training_config.perc_sent_weights
            ),
            model_weight=self.accumulated_weight,
            optimizer_state=self._model.optimizer.variables()
            if self._training_config.serialize_optimizer
            else None,
        )

    def save_model(
        self,
        *,
        latest_weights: ModelWeights,
        best_update_model_weights: ModelWeights,
        time: int,
        best_update_val_loss: Loss,
        updates_without_improving: int,
        new_model_weight: int,
    ) -> None:
        """
        Update the best model and reset the early stopping counter if necessary.

        Update the best model with the received model weights, if it improved the validation loss.
        Also, update the best validation loss achieved so far in that case and reset the early
        stopping counter.

        Otherwise, increase the early stopping counter by one.
        Check if the stop criterion is met and eventually change the node state to STOPPED.

        :param best_update_model_weights: the weights of the best model trained during the last update
        :param latest_weights: the weights to be saved.
        :param time: the current time.
        :param best_update_val_loss: the validation loss achieved by the received weights
        :param updates_without_improving: the number of updates without improvements
        :param new_model_weight: the new model weight
        :return: whether the node has improved the best loss
        """
        self._model.set_weights(latest_weights)
        self.accumulated_weight = new_model_weight

        improvement = self._best_val_loss - best_update_val_loss
        reset_early_stopping = improvement >= self._training_config.min_delta

        if improvement > 0:
            self._logger.debug_log(
                f"Node {self.id} improved loss by {improvement:.4f}. Early stopping is"
                f"{'' if reset_early_stopping else ' not'} reset."
            )
        else:
            self._logger.debug_log(f"Node {self.id} did not improve loss")

        self.update_best_model(best_update_model_weights, best_update_val_loss)

        # if the improvement is greater than min_delta, reset early stopping counter; otherwise,
        # increase it and eventually stop the node if the max number of epochs without improving
        # is reached
        if reset_early_stopping:
            self._updates_without_improving = 0
            self._last_improved_time = time
        else:
            self._updates_without_improving = updates_without_improving + 1
            self._logger.debug_log(
                f"This was the {self._updates_without_improving} update without improvement for node {self.id}"
            )

        satisfied_stop_criterion = self._check_stop_criterion()

        # if self.state != NodeState.FAILED:
        #     self.state = NodeState.STOPPED if satisfied_stop_criterion else NodeState.ACTIVE
        # else:
        #     self.state_before_failure = NodeState.STOPPED if satisfied_stop_criterion else NodeState.ACTIVE

        self.state = NodeState.STOPPED if satisfied_stop_criterion else NodeState.ACTIVE

    def update_best_model(self, weights: ModelWeights, val_loss: float):
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._best_weights = weights

    def persist_best_model(self) -> None:
        """
        Serialize and persist the best model achieved so far.
        """
        best_model = self._create_model()
        best_model.set_weights(self._best_weights)

        best_model.save(
            str(
                self._workspace_dir
                / self._training_config.models_folder
                / f"{self.id}.h5"
            )
        )

    def receive_weights(self, received: WeightsMessage, from_node: NodeId) -> None:
        """
        Receive marshaled weights from a node and store them in the internal buffer.

        :param received: the received weights
        :param from_node: the node from which the weights came from
        """
        self._received_weights[from_node] = received

    def _evaluate(self) -> None:
        if (
            self._history_config.eval_test
            and self._completed_updates % self._history_config.freq == 0
        ):
            X, Y = self._test_set
            pred = self._model.predict(X, verbose=0)

            metrics = compute_metrics(Y, pred)
            self.eval_metrics.append(metrics)

    @property
    def ready_to_train(self) -> bool:
        """
        Whether then node has buffered a minimum number of models to perform an update.
        """
        return len(self._received_weights) >= self._training_config.num_merged_models

    def _check_stop_criterion(self):
        """
        Check if the stop criterion is met and eventually stop the node.
        """
        if self._training_config.stop_criterion == StopCriterion.NO_IMPROVEMENTS:
            satisfied_stop_criterion = (
                self._updates_without_improving >= self._training_config.patience
            )
        elif self._training_config.stop_criterion == StopCriterion.FIXED_UPDATES:
            satisfied_stop_criterion = (
                self._completed_updates >= self._training_config.fixed_updates
            )
        else:
            raise Exception("Unrecognized stop criterion!")
        
        return satisfied_stop_criterion
