import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

DATASETS = ('amt', 'multifc', 'mc_fake', 'jruvika')


def build_graph(x, k=10):
    k = min(k, len(x)-1)
    neighbors = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=2).fit(x)
    distance, indices = neighbors.kneighbors(x, n_neighbors=k+1)
    rows, cols, weights = [], [], []
    for row in range(len(x)):
        valid = indices[row] != row
        selected = indices[row][valid][:k]
        values = np.maximum(1-distance[row][valid][:k], 0)
        rows.extend([row]*len(selected))
        cols.extend(selected)
        weights.extend(values)
    adjacency = sparse.csr_matrix((np.asarray(weights, dtype='float32'), (rows, cols)), shape=(len(x),len(x)))
    adjacency.eliminate_zeros()
    return neighbors, normalize(adjacency, norm='l1', axis=1), k


def query_adjacency(x, neighbors, anchor_count, k):
    distance, indices = neighbors.kneighbors(x, n_neighbors=k)
    weights = np.maximum(1-distance, 0).astype('float32')
    adjacency = sparse.csr_matrix((weights.ravel(), (np.repeat(np.arange(len(x)), k), indices.ravel())), shape=(len(x),anchor_count))
    return normalize(adjacency, norm='l1', axis=1)


def numpy_logits(x, adjacency, anchors, anchor_hidden, state):
    hidden = np.maximum(x @ state['self1.weight'].T + (adjacency @ anchors) @ state['neighbor1.weight'].T + state['self1.bias'], 0)
    return hidden @ state['self2.weight'].T + (adjacency @ anchor_hidden) @ state['neighbor2.weight'].T + state['self2.bias']


def make_network(input_dim, hidden_dim, dropout):

    import torch
    from torch import nn

    class WeightedMeanGraphSAGE(nn.Module):
        def __init__(self):
            super().__init__()
            self.self1 = nn.Linear(input_dim, hidden_dim)
            self.neighbor1 = nn.Linear(input_dim, hidden_dim, bias=False)
            self.self2 = nn.Linear(hidden_dim, 2)
            self.neighbor2 = nn.Linear(hidden_dim, 2, bias=False)
            self.dropout = nn.Dropout(dropout)

        def encode(self, x, neighbor_x):
            return torch.relu(self.self1(x) + self.neighbor1(neighbor_x))

        def forward(self, x, adjacency, neighbor_x):
            hidden = self.dropout(self.encode(x, neighbor_x))
            return self.self2(hidden) + self.neighbor2(torch.sparse.mm(adjacency, hidden))

    return WeightedMeanGraphSAGE()
