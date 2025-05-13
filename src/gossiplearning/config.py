from enum import Enum
from pathlib import Path

from pydantic import (
    BaseModel,
    Field,
    model_validator,
    field_validator,
    field_serializer,
)

from gossiplearning.models import (
    FailureMode,
    Link,
    MergeStrategy,
    StopCriterion,
    NodeId,
)


class LogLevel(int, Enum):
    """Enum containing different log levels for the simulator."""

    DEBUG = 2
    INFO = 1
    ERROR = 0


class NodeConfig(BaseModel):
    """
    Configuration related to a single gossip node.
    """

    id: NodeId = Field(
        ...,
        ge=0,
        description="Numerical ID representing the node, ranging from 0 to the number of nodes ("
        "excluded)",
    )
    links: tuple[Link, ...] = Field(
        ..., description="List of links connecting to neighbors"
    )

    @model_validator(mode="after")
    def validate_node(self) -> "NodeConfig":
        if any(neighbor.node == self.id for neighbor in self.links):
            raise ValueError("A node cannot be its own neighbor")

        if len(self.links) != len(set([neighbor.node for neighbor in self.links])):
            raise ValueError("Node neighbors lists cannot contain duplicates")
        return self


class TrainingConfig(BaseModel):
    """
    Configuration related to training and gossip learning protocol parameters.
    """

    patience: int = Field(
        ...,
        ge=1,
        description="The number of epochs without improvements that are executed by each node "
        "before stopping",
    )
    min_delta: float = Field(
        ...,
        ge=0,
        description="The minimum improvement required to consider an epoch as an improvement",
    )
    perc_sent_weights: float = Field(
        ...,
        gt=0,
        le=1,
        description="The percentage of weights that are sent at each iteration",
    )
    target_probability: float = Field(
        ...,
        gt=0,
        le=1,
        description="The probability that a neighbor node is selected to send model weights",
    )
    failure_mode: FailureMode = Field(
        FailureMode.NONE, description="The failure mode to be used"
    )
    is_time_to_fail_frequency: int = Field(
        ...,
        ge=1,
        description="The frequency at which the simulator checks if it is time to inject a failure",
    )
    node_recovery_time_mean: int = Field(
        ...,
        ge=1,
        description="The mean time it takes for a node to recover from a failure",
    )
    node_recovery_time_std: int = Field(
        ...,
        ge=0,
        description="The standard deviation of the time it takes for a node to recover from a failure",
    )
    node_failure_probability: float = Field(
        ...,
        gt=0,
        le=1,
        description="The probability that a node is failing",
    )
    max_percentage_failed_nodes: float = Field(
        ...,
        ge=0,
        le=1,
        description="The maximum percentage of nodes that can fail at the same time",
    )
    link_failure_probability: float = Field(
        ...,
        gt=0,
        le=1,
        description="The probability that a node is failing",
    )
    models_folder: str = Field(
        ..., description="The path of the folder where models should be saved"
    )
    n_input_features: int = Field(
        ..., description="The number of input features per timestep"
    )
    n_output_vars: int = Field(..., description="The number of predicted variables")
    merge_strategy: MergeStrategy = Field(
        MergeStrategy.SIMPLE_AVG, description="The strategy used to merge model weights"
    )
    batch_size: int = Field(..., description="The batch size to be used for training")
    epochs_per_update: int = Field(
        ...,
        ge=1,
        description="The number of training epochs that are"
        "performed each time a node triggers an "
        "update of the model",
    )
    stop_criterion: StopCriterion = Field(
        ..., description="The stop criterion to be used"
    )
    fixed_updates: int = Field(
        1,
        description="The number of fixed updates to be performed if the "
        "respective stop criterion is chosen",
    )
    num_merged_models: int = Field(
        1, description="The number of model weights to merge each time"
    )
    shuffle_batch: bool = Field(
        True, description="Whether to shuffle batch data during training"
    )
    finetuning_epochs: int = Field(
        0,
        description="The number of fine-tuning epochs that each "
        "node performs at the end of the Gossip "
        "protocol using its own data",
    )
    serialize_optimizer: bool = Field(
        False,
        description="Whether to serialize the optimizer state together with the model weights",
    )


class HistoryConfig(BaseModel):
    eval_test: bool = Field(
        False,
        description="Whether to evaluate node performance on a common test dataset",
    )
    freq: int = Field(
        5, description="How often (updates) node models should be evaluated"
    )


class Config(BaseModel):
    """
    Configuration object to be used for configuring the simulator.
    """

    n_nodes: int = Field(..., description="Number of nodes")
    nodes: tuple[NodeConfig, ...] = Field(())
    training: TrainingConfig = Field(...)
    log_level: LogLevel = Field("INFO", description="The simulator log level")
    workspace_dir: Path = Field(..., description="The workspace directory")
    history: HistoryConfig = Field(..., description="History config")

    @field_validator("workspace_dir")
    @classmethod
    def parse_and_validate_workspace_dir(cls, value: str) -> Path:
        """
        Parse a workspace directory string into a Path object, implicitly validating it.

        :param value: the workspace directory string
        :return: the corresponding path object
        """
        return Path(value)

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        # only validate length of nodes if provided
        if len(self.nodes) > 0 and self.n_nodes != len(self.nodes):
            raise ValueError("Inconsistent number of nodes")

        if any(
            node.id >= self.n_nodes
            or any(neighbor.node >= self.n_nodes for neighbor in node.links)
            for node in self.nodes
        ):
            raise ValueError(
                "Node IDs should be numbers between 0 and N-1, where N is the number of nodes"
            )

        if any(
            len(node_conf.links) < self.training.num_merged_models
            for node_conf in self.nodes
        ):
            raise Exception(
                f"There are nodes with less than {self.training.num_merged_models} neighbors!"
            )

        if (
            self.training.num_merged_models > 1
            and self.training.merge_strategy == MergeStrategy.OVERWRITE
        ):
            raise Exception("Overwrite merge strategy only works with 1 merged model")

        if (
            self.training.merge_strategy == MergeStrategy.OVERWRITE
            and self.training.perc_sent_weights < 1
        ):
            raise Exception(
                "Overwrite merge strategy only works with no weights subsampling"
            )

        return self

    @field_serializer("workspace_dir")
    def serialize_path(self, value: Path) -> str:
        return str(value)
