import os
import subprocess
from utils import get_repo_root, log_data
import logging
import time

# Define the datasets and GNN layers to test
datasets = ['KKI', 'OHSU', 'PROTEINS', 'AIDS']
gnn_layers = ['GCN', 'GAT']
inter_graph_methods = ['graph2vec', 'diff2vec']

# Ensure the required directories exist
def ensure_directories(dataset, gnn_layer):
    repo_root = get_repo_root()
    data_dir = os.path.join(repo_root, 'data', dataset, gnn_layer)
    os.makedirs(data_dir, exist_ok=True)
    log_data(f"Ensured directory exists: {data_dir}")

# Run the experiments
def run_experiments():
    experiment_counter = 0
    for dataset in datasets:
        for gnn_layer in gnn_layers:
            if experiment_counter > 0:
                log_data("Waiting for 15 seconds before the next experiment...")
                time.sleep(15)

            experiment_name = f"[Experiment {experiment_counter + 1} - Dataset: {dataset}, Layer type: {gnn_layer}, Method: diff2vec]"
            log_data(experiment_name)
            ensure_directories(dataset, gnn_layer)
            
            # Construct the command to run main.py
            command = [
                'python', 'src/main.py',
                '--dataset', dataset,
                '--gnn_layer', gnn_layer,
                '--inter_graph_method', 'diff2vec',
            ]
            
            log_data(f"Running command: {' '.join(command)}")
            
            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)
            
            # log_data the output and errors
            log_data(result.stdout)
            if result.stderr:
                log_data(f"Error: {result.stderr}")

            log_data(f"[End of {experiment_name}]")
            experiment_counter += 1
    for dataset in datasets:
        for gnn_layer in gnn_layers:
            if experiment_counter > 0:
                log_data("Waiting for 15 seconds before the next experiment...")
                time.sleep(15)

            experiment_name = f"[Experiment {experiment_counter + 1} - Dataset: {dataset}, Layer type: {gnn_layer}, Method: graph2vec]"
            log_data(experiment_name)
            ensure_directories(dataset, gnn_layer)
            
            # Construct the command to run main.py
            command = [
                'python', 'src/main.py',
                '--dataset', dataset,
                '--gnn_layer', gnn_layer,
                '--inter_graph_method', 'graph2vec',
            ]
            
            log_data(f"Running command: {' '.join(command)}")
            
            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)
            
            # log_data the output and errors
            log_data(result.stdout)
            if result.stderr:
                log_data(f"Error: {result.stderr}")

            log_data(f"[End of {experiment_name}]")
            experiment_counter += 1

if __name__ == "__main__":
    run_experiments()