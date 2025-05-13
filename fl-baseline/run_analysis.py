import csv
import json
import os
from numpy import mean, std
import pandas as pd
import typer

app = typer.Typer()


def _run_analysis(experiments_dir: str, experiment_ids: str, network_name: str, runs: int, scenarios: int, nodes: int, results_summary: str, all_results: str):
    mse = {'generalized': {}, 'classical': {}}
    for id in experiment_ids:
        mse['classical'][id] = []
        mse['generalized'][id] = []
        for s in range(scenarios):
             for r in range(runs):
                with open(f"{experiments_dir}/{id}/{network_name}/{s}/run_{r}/results.json") as f:
                    data = json.load(f)
                    mse['generalized'][id].append(pd.DataFrame(data["centralized_evaluate"])["centralized_mse"].tail(1).values[0])
                nodes_mse= []
                for n in range(nodes):
                    with open(f"{experiments_dir}/{id}/{network_name}/{s}/run_{r}/node_{n}/evaluate_metrics.csv") as f:
                        nodes_mse.append(pd.read_csv(f)['mse'].tail(1).values[0]) # or min
                mse['classical'][id].append(mean(nodes_mse))
        gen_mmse = mean(mse['generalized'][id])
        gen_std = std(mse['generalized'][id])
        gen_rsd = gen_std / gen_mmse

        cls_mmse = mean(mse['classical'][id])
        cls_std = std(mse['classical'][id])
        cls_rsd =  cls_std / cls_mmse

        metrics = {
            'id': id,
            'classical_mmse': f"{cls_mmse:.0f}",
            'classical_std': f"{cls_std:.0f}",
            'classical_rsd': f"{cls_rsd:.2f}",
            'generalized_mmse': f"{gen_mmse:.0f}",
            'generalized_std': f"{gen_std:.0f}",
            'generalized_rsd': f"{gen_rsd:.2f}"
        }

        write_header = not os.path.exists(results_summary)
        with open(results_summary, "a") as file:
            writer = csv.DictWriter(file, fieldnames=['id', 'classical_mmse', 'classical_std', 'classical_rsd', 'generalized_mmse', 'generalized_std', 'generalized_rsd'])
            if write_header:
                writer.writeheader()
            writer.writerow(metrics)
        
        write_header = not os.path.exists(all_results)
        with open(all_results, "a") as file:
            writer = csv.DictWriter(file, fieldnames=['id', 'MSE', 'strategy', 'metric'])
            if write_header:
                writer.writeheader()
            for r in mse['classical'][id]:
                metrics = {
                    'id': id,
                    'MSE': f"{r:.2f}",
                    'strategy': 'Federated Learning',
                    'metric': 'classical'
                }
                writer.writerow(metrics)
            for r in mse['generalized'][id]:
                metrics = {
                    'id': id,
                    'MSE': f"{r:.2f}",
                    'strategy': 'Federated Learning',
                    'metric': 'generalized'
                }
                writer.writerow(metrics)


@app.command("sampling")
def run_sampling_analysis():
    experiments_dir = "experiments/sampling"
    sampling_rates = [25, 50, 75, 100]
    network_name = "porto_10n_3k"
    runs = 10
    scenarios = 10
    nodes = 10
    
    analysis_dir = f"analysis/sampling/{network_name}"
    os.makedirs(analysis_dir, exist_ok=True)

    results_summary = f"{analysis_dir}/results_summary.csv"
    os.remove(results_summary) if os.path.exists(results_summary) else None
    
    all_results = f"{analysis_dir}/results.csv"
    os.remove(all_results) if os.path.exists(all_results) else None

    _run_analysis(experiments_dir, sampling_rates, network_name, runs, scenarios, nodes, results_summary, all_results)

@app.command("epochs-updates")
def run_epochs_updates_analysis():
    experiments_dir = "experiments/epochs-updates"
    epochs_updates = ['1-100', '1-20', '1-50', '2-20', '2-50', '5-20', '5-50']
    network_name = "porto_10n_3k"
    runs = 10
    scenarios = 10
    nodes = 10

    analysis_dir = f"analysis/epochs-updates/{network_name}"
    os.makedirs(analysis_dir, exist_ok=True)

    results_summary = f"{analysis_dir}/results_summary.csv"
    os.remove(results_summary) if os.path.exists(results_summary) else None
    
    all_results = f"{analysis_dir}/results.csv"
    os.remove(results_summary) if os.path.exists(results_summary) else None

    _run_analysis(experiments_dir, epochs_updates, network_name, runs, scenarios, nodes, results_summary, all_results)
    
    

if __name__ == "__main__":
    app()