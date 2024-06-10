import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
import numpy as np

class Graph2Vec(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Graph2Vec, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x.mean(dim=0)

# def generate_graph2vec_embeddings(graphs, gnn_model, device):
#     loader = DataLoader(graphs, batch_size=1)
#     embeddings = []
#     for data in loader:
#         data.to(device)
#         gnn_model.to(device)
#         embedding = gnn_model(data)[2]
#         embeddings.append(embedding.detach().cpu().numpy())
#     embeddings = np.array(embeddings)
#     return embeddings


def generate_graph2vec_embeddings(graphs, model, device):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in graphs:
            data = data.to(device)
            _, _, graph_embedding = model(data)
            embeddings.append(graph_embedding.cpu().numpy().flatten())
    return np.array(embeddings)