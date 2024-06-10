import torch
import torch.nn as nn
import torch.nn.functional as F
from GNN import GmapAD_GCN, GmapAD_GAT
import logging

logging.basicConfig(filename='siamese_gnn_log.log', level=logging.INFO, format='\n[%(asctime)s] %(message)s')

class SiameseGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gnn_type='GCN'):
        super(SiameseGNN, self).__init__()
        if gnn_type == 'GCN':
            self.gnn = GmapAD_GCN(num_nodes=None, input_dim=input_dim, hidden_channels=hidden_dim, num_classes=output_dim)
        else:
            self.gnn = GmapAD_GAT(num_nodes=None, input_dim=input_dim, hidden_channels=hidden_dim, num_classes=output_dim)

    def forward_once(self, x):
        _, _, g_rep = self.gnn(x)
        return g_rep

    def forward(self, x1, x2):
        output1 = self.forward_once(x1)
        output2 = self.forward_once(x2)
        return output1, output2

def train_siamese_gnn(model, siamese_data, optimizer, criterion, device):
    model.train()
    for epoch in range(1, 201):
        for data1, data2, label in siamese_data:
            data1, data2 = data1.to(device), data2.to(device)
            label = torch.tensor([label], dtype=torch.float32).to(device)  # Ensure label is a tensor of the correct size
            optimizer.zero_grad()
            output1, output2 = model(data1, data2)
            output = F.pairwise_distance(output1, output2)
            output = torch.sigmoid(output)  # Apply sigmoid to constrain the output to [0, 1]
            
            # Log and check outputs
            logging.info(f"Output1: {output1}, Output2: {output2}, Distance: {output}")
            assert (output >= 0).all() and (output <= 1).all(), f"Output out of range: {output}"
            
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            logging.info(f"Epoch: {epoch}, Loss: {loss.item()}")

def prepare_siamese_data(graphs, gnn_model):
    data_pairs = []
    for i, graph1 in enumerate(graphs):
        for j, graph2 in enumerate(graphs):
            if i != j:
                label = 1 if graph1.y == graph2.y else 0
                data_pairs.append((graph1, graph2, label))
    return data_pairs

def evaluate_siamese_gnn(model, test_graphs, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for data1, data2, label in test_graphs:
            data1, data2 = data1.to(device), data2.to(device)
            output1, output2 = model(data1, data2)
            distance = F.pairwise_distance(output1, output2)
            distance = torch.sigmoid(distance)  # Apply sigmoid to constrain the output to [0, 1]
            prediction = 1 if distance < 0.5 else 0
            predictions.append(prediction)
            true_labels.append(label)
    return predictions, true_labels