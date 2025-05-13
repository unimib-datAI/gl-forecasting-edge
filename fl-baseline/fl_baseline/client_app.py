"""fl-baseline: A Flower / TensorFlow app."""

import csv
import os
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from fl_baseline.task import create_run_dir, load_data, load_model


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(
        self, model, data, epochs, batch_size, experiments_dir, verbose
    ):
        self.model = model
        self.data = data
        self.epochs = epochs
        self.batch_size = batch_size
        self.fit_metrics = ['loss', 'mse', 'mae', 'msle', 'mape', 'rmse', 'val_loss', 'val_mse', 'val_mae', 'val_msle', 'val_mape', 'val_rmse']
        self.evaluate_metrics = ['mae', 'msle', 'mse', 'mape', 'rmse']
        self.experiments_dir = experiments_dir
        self.verbose = verbose

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        history = self.model.fit(
            self.data['X_train'],
            self.data['Y_train'],
            validation_data=(self.data["X_val"], self.data["Y_val"]),
            shuffle=True,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        results = {
            "loss": history.history["loss"][0],
            "mse": history.history["mse"][0],
            "mae": history.history["mae"][0],
            "msle": history.history["msle"][0],
            "mape": history.history["mape"][0],
            "rmse": history.history["root_mean_squared_error"][0],
            "val_loss": history.history["val_loss"][0],
            "val_mse": history.history["val_mse"][0],
            "val_mae": history.history["val_mae"][0],
            "val_msle": history.history["val_msle"][0],
            "val_mape": history.history["val_mape"][0],
            "val_rmse": history.history["val_root_mean_squared_error"][0],
        }
        self._write_metrics("fit_metrics", self.fit_metrics, results)
        return self.model.get_weights(), len(self.data['X_train']), results

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, mae, msle, mse, mape, rmse = self.model.evaluate(self.data["X_test"], self.data["Y_test"], verbose=0)
        results = {"mae": mae, "msle": msle, "mse": mse, "mape": mape, "rmse": rmse}
        self._write_metrics("evaluate_metrics", self.evaluate_metrics, results)
        return loss, len(self.data["X_test"]), results
    
    def _write_metrics(self, metrics_file, metrics, metric_values):
        metrics_file = f"{self.experiments_dir}/{metrics_file}.csv"
        write_header = not os.path.exists(metrics_file)
        with open(metrics_file, "a") as file:
            writer = csv.DictWriter(file, fieldnames=metrics)
            if write_header:
                writer.writeheader()
            writer.writerow(metric_values)


def client_fn(context: Context):
    # Load model and data
    net = load_model(context.run_config)
    partition_id = context.node_config["partition-id"]
    datasets_dir = context.run_config.get("datasets-dir")
    network_name = context.run_config.get("network-name")
    network_scenario = context.run_config.get("network-scenario")
    dataset_name = context.run_config.get("dataset-name")
    dataset = f"{datasets_dir}/{network_name}/{network_scenario}/{dataset_name}"
    epochs = context.run_config["local-epochs"]
    batch_size = context.run_config["batch-size"]
    verbose = context.run_config.get("verbose")

    data = load_data(partition_id, dataset)

    save_dir, _ = create_run_dir(context.run_config)
    save_dir = save_dir / f"node_{partition_id}"

    os.makedirs(save_dir, exist_ok=True)
    
    # Return Client instance
    return FlowerClient(
        net, data, epochs, batch_size, save_dir, verbose
    ).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)
