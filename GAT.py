import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool
from torch import Tensor
from typing import Optional

class GAT(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, out_channels: int, heads: int = 1, 
                 top_features: int = 4, forward_on_top_features: bool = True, depth: int = 2, 
                 node_classification: bool = False):
        """
        Initializes the GAT model.

        Args:
            node_features (int): Number of input node features.
            hidden_dim (int): Dimension of the hidden layers.
            out_channels (int): Number of output channels.
            heads (int, optional): Number of attention heads in each GAT layer. Defaults to 1.
            top_features (int, optional): Number of additional topological features. Defaults to 4.
            forward_on_top_features (bool, optional): Whether to concatenate topological features to node features. Defaults to True.
            depth (int, optional): Number of GAT layers. Defaults to 2.
            node_classification (bool, optional): Whether the task is node classification or not. Defaults to False.
        """
        super(GAT, self).__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.forward_on_top_features = forward_on_top_features
        self.depth = depth
        self.node_classification = node_classification
        
        if forward_on_top_features:
            self.in_channels = node_features + top_features
        else:
            self.in_channels = node_features

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels=self.in_channels, out_channels=hidden_dim, heads=heads))
        
        for _ in range(1, depth):
            self.convs.append(GATConv(in_channels=heads * hidden_dim, out_channels=hidden_dim, heads=heads))

        self.fc = nn.Linear(hidden_dim * heads, out_channels)

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass of the GAT model.

        Args:
            data (Data): Input data containing edge indices, node features, and other attributes.

        Returns:
            Tensor: Output of the model, either node-level or graph-level predictions.
        """
        edge_index, batch = data.edge_index, data.batch

        if self.forward_on_top_features:
            x = torch.concat((data.x, data.top_features), dim = -1)
        else:
            x = data.x

        for i, layer in enumerate(self.convs):
            if i == (self.depth - 1):
                x = F.dropout(x, p=0.5, training=self.training)

            x = F.relu(layer(x, edge_index))

        # Global mean pooling to create graph-level embeddings
        if not self.node_classification:
            x = global_mean_pool(x, batch)

        # Pass through the final fully connected layer
        x = self.fc(x)

        return x
