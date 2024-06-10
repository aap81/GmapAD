import torch
import torch.nn as nn
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import load_dataset, pos_graphs_pool, print_dataset_stat, get_repo_root, log_data, log_time, arg_parser, downsample, ensure_directories
from GNN import GmapAD_GCN, GmapAD_GAT, train_gnn
from evolution import evolution_svm
from graph2vec import generate_graph2vec_embeddings
from diff2vec import generate_diff2vec_embeddings
from siamese_gnn import SiameseGNN, train_siamese_gnn, evaluate_siamese_gnn, prepare_siamese_data
import os
import time

def main():
    start_time = time.time()
    action_start_time = time.time()
    args = arg_parser()
    log_time(start_time, action_start_time, "Parsed arguments")

    # Check GPU availability
    device_str = 'cpu'
    if torch.cuda.is_available():
        device_str = 'cuda'
    device = torch.device(device_str)
    args.device = device
    ensure_directories(args.dataset, args.gnn_layer)
    log_data(f"Training device: {device}")
    log_data(f"loading dataset {args.dataset}")
    log_data(f"Testing Round: {args.round}")

    # Load or generate dataset
    data_root_path = os.path.join(get_repo_root(), 'data')
    graph_path = f"{data_root_path}/{args.dataset}/{args.gnn_layer}/graph{args.round}.pt"
    train_path = f"{data_root_path}/{args.dataset}/{args.gnn_layer}/train_graph{args.round}.pt"
    val_path = f"{data_root_path}/{args.dataset}/{args.gnn_layer}/val_graph{args.round}.pt"
    test_path = f"{data_root_path}/{args.dataset}/{args.gnn_layer}/test_graph{args.round}.pt"
    
    if not os.path.exists(graph_path) or not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
        graphs = load_dataset(args.dataset, args)
        graphs = graphs.shuffle()
        torch.save(graphs, graph_path)
        train_ratio = args.train_ratio
        val_ratio = args.test_ratio
        test_ratio = 1 - train_ratio - val_ratio
        train_graphs = graphs[:int(len(graphs) * train_ratio)]
        val_graphs = graphs[int(len(graphs) * train_ratio): int(len(graphs) * (train_ratio + val_ratio))]
        test_graphs = graphs[int(len(graphs) * (train_ratio + val_ratio)):]
        train_graphs = downsample(args.ds_rate, args.ds_cl, train_graphs)
        val_graphs = downsample(args.ds_rate, args.ds_cl, val_graphs)
        test_graphs = downsample(args.ds_rate, args.ds_cl, test_graphs)
        torch.save(train_graphs, train_path)
        torch.save(val_graphs, val_path)
        torch.save(test_graphs, test_path)
    else:
        log_data("Load from pre-splitted data.")
        graphs = torch.load(graph_path)
        train_graphs = torch.load(train_path)
        val_graphs = torch.load(val_path)
        test_graphs = torch.load(test_path)

    print_dataset_stat(args, graphs)
    log_time(start_time, action_start_time, "Loaded dataset and Stats")
    action_start_time = time.time()  # training start

    if args.gnn_layer == "GCN":
        model = GmapAD_GCN(num_nodes=graphs[0].x.shape[0], input_dim=graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2)
    else:
        model = GmapAD_GAT(num_nodes=graphs[0].x.shape[0], input_dim=graphs[0].x.shape[1], hidden_channels=args.gnn_dim, num_classes=2, num_heads=args.gat_heads)
    
    model = model.to(device)
    log_data(f"Start training model {args.gnn_layer}")
    train_gnn(model, train_graphs, val_graphs, test_graphs, args)
    log_time(start_time, action_start_time, "Completed model training")

    # Generate Node Pool for Evolution SVM
    if args.inter_graph_method == "evolution":
        pos_graphs = [graph for graph in train_graphs if graph.y == 1]
        node_pool = pos_graphs_pool(pos_graphs, model, args)
        node_pool = node_pool.cpu()
        log_data(f"Generating Node pool size: {node_pool.size()}")

    # Convert labels to CPU and NumPy arrays
    Y_train = np.array([graph.y.cpu().numpy() for graph in train_graphs])
    Y_val = np.array([graph.y.cpu().numpy() for graph in val_graphs])
    Y_test = np.array([graph.y.cpu().numpy() for graph in test_graphs])

    log_data(f"Y_train length: {len(Y_train)}, Y_val length: {len(Y_val)}, Y_test length: {len(Y_test)}")

    # Select inter-graph analysis method
    if args.inter_graph_method == "evolution":
        action_start_time = time.time()  # training start
        clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000)
        log_data(f"Test on {args.dataset}, using Evolution SVM, graph pool is {args.n_p_g}, node pool stg is {args.n_p_stg}")
        log_data(f"Test graph length: {len(test_graphs)}")
        clf, x_train_pred, Y_train, x_val_pred, Y_val, x_test_pred, Y_test = evolution_svm(clf, model, node_pool, args, train_graphs, val_graphs, test_graphs)
        log_time(start_time, action_start_time, "Completed SVM evaluation")
    elif args.inter_graph_method == "graph2vec":
        action_start_time = time.time()  # training start
        input_dim = graphs[0].x.shape[1]
        graph_embeddings = generate_graph2vec_embeddings(graphs, model, device)
        train_size = len(train_graphs)
        val_size = len(val_graphs)
        test_size = len(test_graphs)

        # Align graph embeddings with splits
        X_train = graph_embeddings[:train_size]
        X_val = graph_embeddings[train_size:train_size + val_size]
        X_test = graph_embeddings[train_size + val_size:train_size + val_size + test_size]
        
        log_data(f"X_train length: {len(X_train)}, X_val length: {len(X_val)}, X_test length: {len(X_test)}")
        assert len(X_train) == len(Y_train), f"Length mismatch: X_train={len(X_train)}, Y_train={len(Y_train)}"
        assert len(X_val) == len(Y_val), f"Length mismatch: X_val={len(X_val)}, Y_val={len(Y_val)}"
        assert len(X_test) == len(Y_test), f"Length mismatch: X_test={len(X_test)}, Y_test={len(Y_test)}"
        
        clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000)
        clf.fit(X_train, Y_train)
        x_test_pred = clf.predict(X_test)
        log_time(start_time, action_start_time, "Completed graph2vec evaluation")
    elif args.inter_graph_method == "diff2vec":
        action_start_time = time.time()  # training start
        input_dim = graphs[0].x.shape[1]
        graph_embeddings = generate_diff2vec_embeddings(graphs, model, device)
        train_size = len(train_graphs)
        val_size = len(val_graphs)
        test_size = len(test_graphs)

        # Align graph embeddings with splits
        X_train = graph_embeddings[:train_size]
        X_val = graph_embeddings[train_size:train_size + val_size]
        X_test = graph_embeddings[train_size + val_size:train_size + val_size + test_size]

        log_data(f"X_train length: {len(X_train)}, X_val length: {len(X_val)}, X_test length: {len(X_test)}")
        assert len(X_train) == len(Y_train), f"Length mismatch: X_train={len(X_train)}, Y_train={len(Y_train)}"
        assert len(X_val) == len(Y_val), f"Length mismatch: X_val={len(X_val)}, Y_val={len(Y_val)}"
        assert len(X_test) == len(Y_test), f"Length mismatch: X_test={len(X_test)}, Y_test={len(Y_test)}"

        clf = svm.SVC(kernel='linear', C=1.0, cache_size=1000)
        clf.fit(X_train, Y_train)
        x_test_pred = clf.predict(X_test)
        log_time(start_time, action_start_time, "Completed diff2vec evaluation")
    elif args.inter_graph_method == "siamese":
        action_start_time = time.time()  # training start
        siamese_model = SiameseGNN(input_dim=graphs[0].x.shape[1], hidden_dim=args.gnn_dim, output_dim=1, gnn_type=args.gnn_layer)
        siamese_model = siamese_model.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(siamese_model.parameters(), lr=args.learning_rate)
        siamese_data = prepare_siamese_data(train_graphs, model)
        logging.info(f"Siamese data prepared: {len(siamese_data)} pairs")
        train_siamese_gnn(siamese_model, siamese_data, optimizer, criterion, device)
        x_test_pred, Y_test = evaluate_siamese_gnn(siamese_model, test_graphs, device)
        log_time(start_time, action_start_time, "Completed siamese evaluation")

    # Compute and log metrics for the selected method
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

if __name__ == "__main__":
    main()