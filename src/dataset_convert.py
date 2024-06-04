import os
import pandas as pd
import numpy as np

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return [line.strip() for line in lines]

def convert_to_nel(dataset_name, input_dir, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f'{dataset_name}.nel')
    
    # Read all required files
    adjacency_list = read_file(os.path.join(input_dir, f'{dataset_name}_A.txt'))
    graph_indicators = read_file(os.path.join(input_dir, f'{dataset_name}_graph_indicator.txt'))
    graph_labels = read_file(os.path.join(input_dir, f'{dataset_name}_graph_labels.txt'))
    node_labels = read_file(os.path.join(input_dir, f'{dataset_name}_node_labels.txt'))

    with open(output_file, 'w') as out_file:
        # Write graph indicators and graph labels
        for i, graph_id in enumerate(graph_indicators):
            node_label = node_labels[i] if i < len(node_labels) else 'unknown'
            out_file.write(f'n {i+1} {node_label}\n')

        # Write adjacency matrix
        for line in adjacency_list:
            row, col = line.split(',')
            out_file.write(f'e {row} {col} 1\n')  # Assuming weight of 1 for all edges

        # Write graph labels
        for i, graph_label in enumerate(graph_labels):
            out_file.write(f'g {i+1} {graph_label}\n')

# Example usage
dataset_name = "OHSU"
root_path = "F:\\workspace\\GmapAD\\data\\OHSU"
output_file = root_path+"\\output"

convert_to_nel(dataset_name, root_path, output_file)