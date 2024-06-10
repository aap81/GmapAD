import os
import time
import logging
import argparse
import torch
from torch_geometric.datasets import TUDataset
import torch.nn.functional as F

logging.basicConfig(filename='script_log.log', level=logging.INFO, format='\nSTRT- %(message)s')

def get_repo_root():
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while not os.path.isdir(os.path.join(current_dir, '.git')):
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir == current_dir:
            raise FileNotFoundError("Could not find the root directory of the repository")
        current_dir = parent_dir
    return current_dir

def milliseconds_to_seconds(milliseconds):
    seconds = milliseconds / 1000
    return f"{seconds:.2f} seconds"

def log_data(text):
    print(text)
    logging.info(text)

def log_time(start_time, action_start_time, message):
    current_time = time.time()
    elapsed_time_since_start = (current_time - start_time) * 1000  # Convert to milliseconds
    elapsed_time_since_last = (current_time - action_start_time) * 1000  # Convert to milliseconds
    current_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_data(f"[TIMER: {current_time_str} | {message} | Time_taken_since_start: {elapsed_time_since_start:.2f} ms ({milliseconds_to_seconds(elapsed_time_since_start)}) | Time_taken_since_last: {elapsed_time_since_last:.2f} ms ({milliseconds_to_seconds(elapsed_time_since_last)})]")

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='OHSU', help="['KKI', 'OHSU', 'MUTAG', 'Mutagenicity', 'PROTEINS', 'AIDS', 'NCI1', 'IMDB-BINARY', 'REDDIT-BINARY')")
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

    # Inter-Graph Analysis Method
    parser.add_argument('--inter_graph_method', type=str, default='evolution', help="['evolution', 'graph2vec', 'diff2vec', 'siamese']")
    
    args = parser.parse_args()
    return args

def downsample(ds_rate, ds_cl, graphs):
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

def ensure_directories(dataset, gnn_layer):
    repo_root = get_repo_root()
    data_dir = os.path.join(repo_root, 'data', dataset, gnn_layer)
    os.makedirs(data_dir, exist_ok=True)
    log_data(f"Ensured directory exists: {data_dir}")

def load_dataset(dataset, args):
    root_path = os.path.join(get_repo_root(), 'data')
    data = TUDataset(root=f'{root_path}/TUDataset', name=f'{dataset}')
    return data

def print_dataset_stat(args, graphs):
    a_graphs, a_nodes, a_edges, t_nodes, t_edges = 0, 0, 0, 0, 0
    for graph in graphs:
        t_nodes += graph.num_nodes
        t_edges += graph.num_edges
        if graph.y == args.ds_cl:
            a_graphs += 1
            a_nodes += graph.num_nodes
            a_edges += graph.num_edges
    
    log_data(f"Graph Statistics {args.dataset} : ")
    log_data("{:<8} | {:<10} | {:<10} | {:<10} ".format("Class", "#Graphs", "Avg. V", "Avg. E" ))
    log_data("{:<8} | {:<10} | {:<10} | {:<10} ".format("G_0", a_graphs, a_nodes/a_graphs, a_edges/a_graphs ))
    log_data("{:<8} | {:<10} | {:<10} | {:<10} ".format("G_1", len(graphs) - a_graphs,(t_nodes - a_nodes) /(len(graphs) - a_nodes),  (t_edges - a_edges) /(len(graphs) - a_graphs) ))

def pos_graphs_pool(graphs, model, args):
    '''
    This function returns the node pool, G0 ... GN are all from positive graphs in the training set.
    '''
    n_reps = []
    g_reps = []
    for graph in graphs:
        graph.to(args.device)
        _, n_rep, g_rep = model(graph)
        n_reps.append(n_rep.detach())
        g_reps.append(g_rep.detach())
    node_pool = n_reps[0]
    for n_rep in n_reps[1:]:
        node_pool = torch.cat((node_pool, n_rep), dim=0)
    g_reps = torch.stack(g_reps)
    g_reps = g_reps.squeeze()
    node_pool = top_k_nodes(node_pool, g_reps, args)
    return node_pool

def top_k_nodes(node_pool, g_reps, args):
    node_pool = F.normalize(node_pool, p=2, dim=-1)
    g_reps = F.normalize(g_reps, p=2, dim=-1)
    cos_sim = torch.matmul(g_reps, node_pool.T)
    node_sims = cos_sim.sum(dim=0)
    top_k_nodes = torch.topk(node_sims, args.topk, -1, True).indices
    node_pool = node_pool.index_select(0, top_k_nodes)

    return node_pool

def class_wise_loss(pred, y):
    criterion = torch.nn.CrossEntropyLoss()
    mask_0 = [False for _ in range(y.shape[0])]
    mask_1 = [False for _ in range(y.shape[0])]
    for i, l in enumerate(y):
        if l == 1:
            mask_1[i] = True
            mask_0[i] = False
        else:
            mask_1[i] = False
            mask_0[i] = True

    loss = criterion(pred[mask_0], y[mask_0]) + criterion(pred[mask_1], y[mask_1])  # Compute the loss.
    return loss

def base_map(g_reps, pool_candidate):
    '''
    input: list of graph representations, node pool candidates
    output: each graph's representation on candidate pool
    '''
    rep = torch.stack(g_reps)
    return torch.cdist(rep, pool_candidate, p=1)