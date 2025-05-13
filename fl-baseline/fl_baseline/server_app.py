"""fl-baseline: A Flower / TensorFlow app."""

from fl_baseline.strategy import CustomFedAvg
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fl_baseline.task import get_common_test_set, load_model
import numpy as np

def gen_evaluate_fn(run_config, x_test, y_test):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        model = load_model(run_config)
        model.set_weights(parameters_ndarrays)
        loss, mae, msle, mse, mape, rmse = model.evaluate(x_test, y_test, verbose=0)
        return loss, {"centralized_mse": mse, 'centralized_mae': mae, 'centralized_msle': msle, 'centralized_mape': mape, 'centralized_rmse': rmse}

    return evaluate

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    datasets_dir = context.run_config.get("datasets-dir")
    network_name = context.run_config.get("network-name")
    network_scenario = context.run_config.get("network-scenario")
    dataset_name = context.run_config.get("dataset-name")

    # Initialize model parameters
    ndarrays = load_model(context.run_config).get_weights()
    parameters = ndarrays_to_parameters(ndarrays)

    # Prepare dataset for central evaluation
    x_test, y_test = get_common_test_set(
        f"{datasets_dir}/{network_name}/{network_scenario}/{dataset_name}",
        context.run_config['num-clients'],
        context.run_config["test-set-percentage"],
    )

    # Define strategy
    strategy = CustomFedAvg(
        run_config=context.run_config,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        initial_parameters=parameters,
        evaluate_fn=gen_evaluate_fn(context.run_config, x_test, y_test),
        # evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)