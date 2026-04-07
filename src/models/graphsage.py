import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GraphSAGE(torch.nn.Module):
    """
    GraphSAGE model architecture.
    Aggregates neighbor features distinctly from central node features,
    making it significantly more robust in heterophilic graph environments.
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        # Using SAGE aggregation which handles feature mixing better
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Classification head
        x = self.lin(x)
        return x
