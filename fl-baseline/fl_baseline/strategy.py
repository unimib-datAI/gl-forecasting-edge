
import json
from logging import INFO
import sys

from fl_baseline.task import create_run_dir, load_model, save_run_config

from flwr.common import logger, parameters_to_ndarrays
from flwr.common.typing import UserConfig
from flwr.server.strategy import FedAvg

class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy: (1) saves results to the filesystem, (2) saves a
    checkpoint of the global  model when a new best is found
    """

    def __init__(self, run_config: UserConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.run_config = run_config
        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir(run_config)
        save_run_config(run_config, self.save_path)

        # Keep track of best acc
        self.best_mse_so_far = sys.maxsize

        # A dictionary to store results as they come
        self.results = {}

    def _store_results(self, tag: str, results_dict):
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_mse(self, round, mse, parameters):
        """Determines if a new best global model has been found.

        If so, the model checkpoint is saved to disk.
        """
        if mse < self.best_mse_so_far:
            self.best_mse_so_far = mse
            logger.log(INFO, "New best global model found: %f", mse)
            
            ndarrays = parameters_to_ndarrays(parameters)
            model = load_model(self.run_config)
            model.set_weights(ndarrays)
            file_name = (self.save_path / f"best_global_model.h5")
            model.save(file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict):
        """A helper method that stores results."""
        # Store results
        self._store_results(
            tag=tag,
            results_dict={"round": server_round, **results_dict},
        )

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central mse is found
        self._update_best_mse(server_round, metrics["centralized_mse"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics