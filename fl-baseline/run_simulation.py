import typer
from sh import flwr

def _process_output(line):
        print(line)

import typer

app = typer.Typer()


@app.command("sampling")
def run_sampling_exp(scenarios: int = 10, runs: int = 10, start_from_scenario: int = 0, start_from_run: int = 0, experiments_dir: str = "experiments", sampling_percentage: float = 0.5):
    for i in range(start_from_scenario, scenarios):
        for j in range(start_from_run, runs):
            r = flwr.run(".", "--run-config", f"network-scenario={i} simulation-run={j} experiments-dir='{experiments_dir}' fraction-fit={sampling_percentage}", _out=_process_output, _bg=True, _err=_process_output)
            r.wait()
        start_from_run = 0


@app.command("epochs-updates")
def run_epochs_updates_exp(scenarios: int = 10, runs: int = 10, start_from_scenario: int = 0, start_from_run: int = 0, epochs: int = 1, updates: int = 100):
    experiments_dir = f"experiments/epochs-updates/{epochs}-{updates}"
    for i in range(start_from_scenario, scenarios):
        for j in range(start_from_run, runs):
            r = flwr.run(".", "--run-config", f"network-scenario={i} simulation-run={j} experiments-dir='{experiments_dir}' local-epochs={epochs} num-server-rounds={updates}", _out=_process_output, _bg=True, _err=_process_output)
            r.wait()
        start_from_run = 0


if __name__ == "__main__":
    app()