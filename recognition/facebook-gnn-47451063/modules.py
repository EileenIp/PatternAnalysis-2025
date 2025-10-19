import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def embed(self, data):
        x, edge_index = data.x, data.edge_index
        return F.relu(self.conv1(x, edge_index))

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6, heads=8):
        super().__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_dim*heads, output_dim, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        hidden = F.dropout(x, p=self.dropout, training=self.training)
        hidden = F.elu(self.conv1(hidden, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        out = self.conv2(hidden, edge_index)
        return out
    
    def embed(self, data):
        x, edge_index = data.x, data.edge_index
        return F.elu(self.conv1(x, edge_index))
    
class SAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, normalize=True)
        self.conv2 = SAGEConv(hidden_dim, output_dim, normalize=True)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        hidden = F.dropout(x, p=self.dropout, training=self.training)
        hidden = F.relu(self.conv1(hidden, edge_index))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        out = self.conv2(hidden, edge_index)
        return out
    
    def embed(self, data):
        x, edge_index = data.x, data.edge_index
        return F.relu(self.conv1(x, edge_index))