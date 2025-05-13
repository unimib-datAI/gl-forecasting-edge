import numpy as np

from gossiplearning.models import Dataset


def weight_by_dataset_size(dataset: Dataset) -> int:
    return len(dataset["X_train"])


def weight_by_requests(dataset: Dataset) -> int:
    return round(np.sum(dataset["Y_train"]) / len(dataset["Y_train"]))
