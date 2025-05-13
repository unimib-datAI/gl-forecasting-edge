import math
from typing import Callable

import keras
import numpy as np

from gossiplearning.models import MarshaledWeights, FlattenedWeights, ModelWeights

MarshalWeightsFn = Callable[[keras.Model, float], MarshaledWeights]


def flatten_weights(model: keras.Model) -> FlattenedWeights:
    num_weights = model.count_params()
    weights = np.empty(num_weights)

    start = 0
    for l in model.get_weights():
        flattened = l.flatten()
        end = start + len(flattened)
        weights[start:end] = flattened
        start = end

    return weights


def marshal_weights_with_random_subsampling(
    model: keras.Model, perc_weights: float = 1
) -> MarshaledWeights:
    flattened_weights = flatten_weights(model)

    n_weights = len(flattened_weights)
    indices = np.arange(0, n_weights)

    if perc_weights == 1:
        return MarshaledWeights(
            indices=indices,
            weights=flattened_weights,
        )

    n_samples = np.int64(perc_weights * n_weights)
    selected_indices = np.random.choice(indices, size=n_samples, replace=False)

    return MarshaledWeights(
        indices=selected_indices,
        weights=flattened_weights[selected_indices],
    )


def unflatten_weights(
    model_blueprint: keras.Model, flattened: FlattenedWeights
) -> ModelWeights:
    """
    Transform a set of flattened weights into proper layer-wise model weights.
    :param flattened: the flattened weights
    :return: the model weights group by layer
    """
    res: ModelWeights = []

    start = 0
    for l in model_blueprint.get_weights():
        end = start + math.prod(l.shape)
        weights = flattened[start:end]
        weights = weights.reshape(l.shape)

        res = res + [weights]

        start = end

    return res
