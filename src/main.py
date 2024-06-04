import argparse
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix
import torch
import numpy as np
import sys
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import load_dataset, pos_graphs_pool, print_dataset_stat, get_repo_root, log_data, log_time, milliseconds_to_seconds
from GNN import GmapAD_GCN, GmapAD_GAT, train_gnn
from evolution import evolution_svm
import os
import random
import logging
import time
def warn(*args, **kwargs):
    pass
import warnings

warnings.warn = warn
gpu_execution_enabled = False


# Get the root directory of the repository
repo_root = get_repo_root()

# Construct the path to the data directory
data_root_path = os.path.join(repo_root, 'data')

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='OHSU', help="['KKI', 'OHSU', 'MUTAG', 'Mutangenicity', 'PROTEINS', 'AIDS', 'NCI1', 'IMDB-BINARY', 'REDDIT-BINARY')")
    parser.add_argument('--ds_rate', type=float, default=0.1, help='Dataset downsampling rate for Graph classification datasets.')
    parser.add_argument('--ds_cl', type=int, default=0, help='The default downsampled class.')

    # GNN related parameters
    parser.add_argument('--gnn_layer', type=str, default='GCN', help="['GCN','GAT']")
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--learning_rate', default=0.005, help='Learning rate of the optimiser.')
    parser.add_argument('--weight_decay', default=5e-4, help='Weight decay of the optimiser.')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--test_ratio', type=float, default=0.1)
    parser.add_argument('--tol', type=int, default=300)
    parser.add_argument('--early_stop', type=bool, default=True, help="Early stop when training GNN")

    # Node pooling related parameters
    parser.add_argument('--n_p_g', type=str, default='positive', help="['positive', 'negative']")
    parser.add_argument('--n_p_stg', type=str, default='mean', help="['mean','max', 'min']")
    
    # Evolving related parameters
    parser.add_argument('--w_stg', type=str, default='one-hot', help="['one-hot']")
    parser.add_argument('--clf', type=str, default='svm', help="['svm', 'others']")
    parser.add_argument('--mut_rate', type=float, default=0.5, help="['svm','nb', 'others']")
    parser.add_argument('--cros_rate', type=float, default=0.9, help="['svm','nb', 'others']")
    parser.add_argument('--evo_gen', type=int, default=2000, help="number of evolution generations")
    parser.add_argument('--cand_size', type=int, default=30, help="candidates in each generation")

    # Model hyperparameters
    parser.add_argument('--gnn_dim', type=int, default=128)
    parser.add_argument('--fcn_dim', type=int, default=32)
    parser.add_argument('--gce_q', default=0.7, help='gce q')
    parser.add_argument('--alpha', type=float, default=1.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--topk', type=int, default=64, help="number of the most informative nodes, this parameter also decides the finally graph embedding dimension.")

    # For GAT only, num of attention heads
    parser.add_argument('--gat_heads', default=8, help='GAT heads')

    # Test round
    parser.add_argument('--round', type=int, default=1, help='test round')

    args = parser.parse_args()
    return args

def downsample(ds_rate, ds_cl, graphs):
    ds_rate = args.ds_rate
    ds_cl = args.ds_cl
    ds_graphs = []
    all_graphs = []
    num_nodes = 0
    for graph in graphs:
        num_nodes += graph.num_nodes
        if graph.y == ds_cl:
            ds_graphs.append(graph)
        all_graphs.append(graph)
    ds_graphs = ds_graphs[int(len(ds_graphs)*ds_rate):]
    [all_graphs.remove(graph) for graph in ds_graphs]
    return all_graphs
    # if args.dataset not in ['KKI', 'OHSU']:
    #     ds_rate = args.ds_rate
    #     ds_cl = args.ds_cl
    #     ds_graphs = []
    #     all_graphs = []
    #     num_nodes = 0
    #     for graph in graphs:
    #         num_nodes += graph.num_nodes
    #         if graph.y == ds_cl:
    #             ds_graphs.append(graph)
    #         all_graphs.append(graph)
    #     ds_graphs = ds_graphs[int(len(ds_graphs)*ds_rate):]
    #     [all_graphs.remove(graph) for graph in ds_graphs]
    #     return all_graphs
    # else:
    #     return graphs

if __name__ == "__main__":
    start_time = time.time()
    action_start_time = time.time()
    args = arg_parser()
    log_time(start_time, action_start_time, "Parsed arguments")
    # Check GPU availability
    device_str = 'cpu'
    if gpu_execution_enabled:
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    args.device = device
    log_data(f"Training device: {device}")
    log_data(f"loading dataset {args.dataset}")
    log_data(f"Testing Round: {args.round}")

    graph_path = f"{data_root_path}/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt"
    train_path = f"{data_root_path}/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt"
    val_path = f"{data_root_path}/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt"
    test_path = f"{data_root_path}/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt"
    action_start_time = time.time() # for loading datasets
    if not os.path.exists(graph_path) or not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
        graphs = load_dataset(args.dataset, args)
        # if False and args.dataset in ['KKI', 'OHSU']:
        #     random.shuffle(graphs)
        # else:
        #     graphs = graphs.shuffle()
        graphs = graphs.shuffle()
        torch.save(graphs, f"{data_root_path}/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt")
        train_ratio = args.train_ratio
        val_ratio = args.test_ratio
        train_graphs = graphs[:int(len(graphs)*train_ratio)]
        val_graphs = graphs[int(len(graphs)*train_ratio): int(len(graphs)*(train_ratio+val_ratio))]
        test_graphs = graphs[int(len(graphs)*(train_ratio+val_ratio)):]

        # Downsampling
        train_graphs = downsample(args.ds_rate, args.ds_cl, train_graphs)
        val_graphs = downsample(args.ds_rate, args.ds_cl, val_graphs)
        test_graphs = downsample(args.ds_rate, args.ds_cl, test_graphs)
        
        # Save downsampled datasets
        torch.save(train_graphs, f"{data_root_path}/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt")
        torch.save(val_graphs, f"{data_root_path}/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt")
        torch.save(test_graphs, f"{data_root_path}/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt")
    else:
        log_data("load from pre-splitted data.")
        graphs = torch.load(f"{data_root_path}/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt")
        train_graphs = torch.load(f"{data_root_path}/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt")
        val_graphs = torch.load(f"{data_root_path}/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt")
        test_graphs = torch.load(f"{data_root_path}/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt")

    print_dataset_stat(args, graphs)
    log_time(start_time, action_start_time, "Loaded dataset and Stats")
    if args.gnn_layer == "GCN":
        model = GmapAD_GCN(num_nodes=graphs[0].x.shape[0], input_dim=graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2)
    else:
        model = GmapAD_GAT(num_nodes=graphs[0].x.shape[0], input_dim=graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2, num_heads=args.gat_heads)
    
    model = model.to(device)
    action_start_time = time.time() # training start
    log_data(f"Start training model {args.gnn_layer}")
    train_gnn(model, train_graphs, val_graphs, test_graphs, args)
    log_time(start_time, action_start_time, "Completed model training")

    # Get the candidate pool, grpah reprsentations
    pos_graphs = []
    neg_graphs = []

    for graph in train_graphs:
        if graph.y == 1:
            pos_graphs.append(graph)
        else:
            neg_graphs.append(graph)

    node_pool = pos_graphs_pool(pos_graphs, model, args)
    node_pool = node_pool.cpu()
    log_data(f"Generating Node pool size: {node_pool.size()}")

    if args.clf == "svm":
        action_start_time = time.time() # Starting SVM evaluation
        clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000)
        log_data(f"Test on {args.dataset}, using SVM, graph pool is {args.n_p_g}, node pool stg is {args.n_p_stg}")
        log_data(f"Test graph lenght: {len(test_graphs)}")
        clf, x_train_pred, Y_train, x_val_pred, Y_val, x_test_pred, Y_test = evolution_svm(clf, model, node_pool, args, train_graphs, val_graphs, test_graphs)
        log_time(start_time, action_start_time, "Completed SVM evaluation")
        # Compute metrics
        accuracy = accuracy_score(Y_test, x_test_pred)
        precision = precision_score(Y_test, x_test_pred)
        recall = recall_score(Y_test, x_test_pred)
        f1 = f1_score(Y_test, x_test_pred)
        cm = confusion_matrix(Y_test, x_test_pred)
        
        log_data(f"Accuracy: {accuracy}")
        log_data(f"Precision: {precision}")
        log_data(f"Recall: {recall}")
        log_data(f"F1 Score: {f1}")
        log_data("Confusion Matrix:")
        log_data(cm)