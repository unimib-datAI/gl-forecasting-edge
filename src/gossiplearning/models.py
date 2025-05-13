from dataclasses import dataclass
from enum import Enum
from typing import Callable, TypedDict

import keras
import numpy as np
import tensorflow as tf
from pydantic import BaseModel, Field

ModelWeights = list[np.ndarray]
FlattenedWeights = np.ndarray
Indices = np.ndarray
Loss = float
NodeId = int
MetricValue = float
MetricName = str
ModelBuilder = Callable[[], keras.Model]
Time = int
Features = np.ndarray
Labels = np.ndarray
LabelledData = tuple[Features, Labels]
TimeFn = Callable[[NodeId, NodeId], Time]


class Dataset(TypedDict):
    X_train: Features
    Y_train: Labels
    X_val: Features
    Y_val: Labels
    X_test: Features
    Y_test: Labels


NodeDataFn = Callable[[NodeId], Dataset]
NodeWeightFn = Callable[[Dataset], int]


@dataclass
class MarshaledWeights:
    """
    Marshaled model weights, as they can be sent from one node to another.

    The included weights may not include all the model weights, thus indices must be provided too.
    """

    # number_of_trained_samples: int
    indices: np.ndarray
    weights: np.ndarray


@dataclass
class WeightsMessage:
    marshaled_weights: MarshaledWeights
    model_weight: int
    optimizer_state: list[tf.Variable] | None


AggregatorFn = Callable[
    [keras.Model, int, tuple[WeightsMessage, ...]], tuple[keras.Model, int]
]


class Link(BaseModel):
    """
    Class representing a link between two gossip nodes and its network features.
    """

    node: NodeId = Field(..., ge=0)
    weights_transmission_time: int = Field(..., gt=0)
    round_trip_time: int = Field(..., gt=0)

class FailureMode(str, Enum):
    """
    The failure mode that can be applied to the gossip learning network.
    """

    NONE = "none"
    LINK_FAILURE = "link_failure"
    NODE_FAILURE = "node_failure"

class MergeStrategy(str, Enum):
    """
    Merging strategy that can be applied by an aggregator between weights of different models.
    """

    SIMPLE_AVG = "simple_avg"
    AGE_WEIGHTED = "age_weighted"
    OVERWRITE = "overwrite"
    IMPROVED_OVERWRITE = "improved_overwrite"


class StopCriterion(str, Enum):
    """
    Stop criterion that can be used to end the gossip protocol.
    """

    NO_IMPROVEMENTS = "no_improvements"
    FIXED_UPDATES = "fixed_updates"


class SendModelStrategy(str, Enum):
    """
    Strategy that can be applied by nodes when deciding to which other nodes they will send weights.
    """

    ALL = "all"
    NEIGHBORS = "neighbors"
