import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6):
        super().__init__()
        # First graph convolution layer maps input features to hidden representation
        self.conv1 = GCNConv(input_dim, hidden_dim)
        # Second graph convolution layer maps hidden representation to output logits
        self.conv2 = GCNConv(hidden_dim, output_dim)
        # Dropout probability used during training for regularisation
        self.dropout = dropout

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        # First GCN layer with nonlinearity
        hidden = F.relu(self.conv1(x, edge_index))
        # Apply dropout to the hidden representation
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        # Second GCN layer with no activation, outputs logits
        out = self.conv2(hidden, edge_index)
        return out
    
    def embed(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        return F.relu(self.conv1(x, edge_index))

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6, heads=8):
        super().__init__()
        # First layer with multiple heads
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
        # Second layer with a single head for output logits
        self.conv2 = GATConv(hidden_dim*heads, output_dim, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        # Input dropout as in the original GAT paper
        hidden = F.dropout(x, p=self.dropout, training=self.training)
        # Add first GAT layer with ELU activation
        hidden = F.elu(self.conv1(hidden, edge_index))
        # Apply dropout on the hidden representation
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        # Apply final GAT layer to produce logits
        out = self.conv2(hidden, edge_index)
        return out
    
    def embed(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        return F.elu(self.conv1(x, edge_index))
    
class SAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6):
        super().__init__()
        # First layer maps to hidden representation
        self.conv1 = SAGEConv(input_dim, hidden_dim, normalize=True)
        # Second SAGE layer outputs logits
        self.conv2 = SAGEConv(hidden_dim, output_dim, normalize=True)
        self.dropout = dropout

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        # Apply dropout to input features
        hidden = F.dropout(x, p=self.dropout, training=self.training)
        # Apply first layer with ReLU nonlinearity
        hidden = F.relu(self.conv1(hidden, edge_index))
        # Further apply dropout on hidden activations
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        # Add second layer for logits
        out = self.conv2(hidden, edge_index)
        return out
    
    def embed(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        return F.relu(self.conv1(x, edge_index))