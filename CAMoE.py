from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class CAMoE_GNN_Layer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, gate_channels: int = 4, experts: int = 3, 
                 decay_lim: int = 200, final_temperature: float = 1.0, gnn_layer: str = 'GCN') -> None:
        """
        Initializes the CAMoE_GNN_Layer.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            heads (int): Number of attention heads (for GAT layers). Default is 1.
            gate_channels (int): Number of channels for the gating network. Default is 4.
            experts (int): Number of expert layers. Default is 3.
            decay_lim (int): Number of batches over which the temperature decays. Default is 200.
            final_temperature (float): Final temperature for the softmax in the gating network. Default is 1.0.
            gnn_layer (str): Type of GNN layer to use ('GAT' or 'GCN'). Default is 'GCN'.
        """        
        super(CAMoE_GNN_Layer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads

        if gnn_layer == 'GAT':
            self.layers = nn.ModuleList(
                [GATConv(in_channels=in_channels, out_channels=out_channels, heads=heads) for _ in range(experts)]
                )
            
        elif gnn_layer == 'GCN': 
            self.layers = nn.ModuleList(
                [GCNConv(in_channels=in_channels, out_channels=out_channels) for _ in range(experts)]
                )

        else:
            raise Exception("Unsupported GNN Layer entered")
            
        self.gating_network = nn.Linear(gate_channels, len(self.layers), bias=False)
        self.decay_lim = decay_lim
        self.final_temperature = final_temperature
        self.batch = 0

        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, gate_features: torch.Tensor, return_gate_res: bool = False):
        """
        Forward pass through the CAMoE_GNN_Layer.
        
        Args:
            x (torch.Tensor): Input feature matrix.
            edge_index (torch.Tensor): Edge indices.
            gate_features (torch.Tensor): Features used by the gating network.
            return_gate_res (bool): Whether to return gating network results. Default is False.
        
        Returns:
            torch.Tensor: Output feature matrix.
            torch.Tensor (optional): Gating network results if return_gate_res is True.
        """
        
        # Calculate temperature
        if self.batch < self.decay_lim:
            temperature = 100 - self.batch / (self.decay_lim * 0.01) + self.final_temperature
        else:
            temperature = self.final_temperature

        self.batch += 1

        # Perform gating
        gate_output = self.gating_network(gate_features)
        gate_output = F.softmax(gate_output / temperature, dim=-1)
        
        # Sum over experts
        out = torch.zeros(x.size(0), self.out_channels * self.heads).to(x.device)

        for i, expert in enumerate(self.layers):
            expert_output = F.relu(expert(x, edge_index))
            out += gate_output[:, i].unsqueeze(1) * expert_output

        if return_gate_res:
            return out, gate_output
        
        return out
    
class CAMoE_GNN(nn.Module):
    def __init__(self, node_features: int, hidden_dim: int, out_channels: int, top_features: int = 4,
                 gate_top_only: bool = True, forward_on_top_features: bool = False, depth: int = 2,
                 final_temperature: float = 1.0, heads: int = 1, residual: bool = False, gnn_layer: str = 'GCN',
                 node_classification: bool = False, decay_lim: int = 200, experts: int = 3) -> None:
        super(CAMoE_GNN, self).__init__()
        """
        Initializes the CAMoE_GNN model.
        
        Args:
            node_features (int): Number of input node features.
            hidden_dim (int): Dimension of hidden layers.
            out_channels (int): Number of output channels.
            top_features (int): Number of topological features. Default is 4.
            gate_top_only (bool): Whether to use only topological features for gating. Default is True.
            forward_on_top_features (bool): Whether to concatenate topological features to input features. Default is False.
            depth (int): Depth of the network (number of layers). Default is 2.
            final_temperature (float): Final temperature for the softmax in the gating network. Default is 1.0.
            heads (int): Number of attention heads (for GAT layers). Default is 1.
            residual (bool): Whether to use residual connections. Default is False.
            gnn_layer (str): Type of GNN layer to use ('GAT' or 'GCN'). Default is 'GCN'.
            node_classification (bool): Whether the task is node classification. Default is False.
            decay_lim (int): Number of batches over which the temperature decays. Default is 200.
            experts (int): Number of expert layers. Default is 3.
        """

        self.forward_on_top_features = forward_on_top_features
        self.gate_top_only = gate_top_only
        self.node_classification = node_classification
        self.depth = depth
        self.heads = heads
        self.residual = residual
        self.gnn_layer = gnn_layer
        self.decay_lim = decay_lim

        if gnn_layer != 'GAT':
            self.heads = 1

        if forward_on_top_features:
            self.in_channels = node_features + top_features
        else:
            self.in_channels = node_features

        if gate_top_only:
            self.gate_channels = top_features
        else:
            self.gate_channels = self.in_channels

        self.convs = nn.ModuleList()
        self.convs.append(CAMoE_GNN_Layer(in_channels=self.in_channels, out_channels=hidden_dim, gate_channels=self.gate_channels, 
                                          final_temperature=final_temperature, heads=self.heads, gnn_layer = self.gnn_layer, 
                                          decay_lim = self.decay_lim, experts = experts))

        for _ in range(1, depth):
            self.convs.append(CAMoE_GNN_Layer(in_channels=hidden_dim * self.heads, out_channels=hidden_dim, gate_channels=self.gate_channels,
                                              final_temperature=final_temperature, heads=self.heads, gnn_layer = self.gnn_layer,
                                              decay_lim = self.decay_lim, experts = experts))

        self.fc = nn.Linear(self.heads * hidden_dim, out_channels)

    def forward(self, data: Data, return_gate_res: bool = False, gate_num_return: int = 0):
        """
        Forward pass through the CAMoE_GNN model.
        
        Args:
            data (Data): Input data object.
            return_gate_res (bool): Whether to return gating network results. Default is False.
            gate_num_return (int): Layer number to return gating results from. Default is 0.
        
        Returns:
            torch.Tensor: Output feature matrix.
            torch.Tensor (optional): Gating network results if return_gate_res is True.
        """
        edge_index, batch = data.edge_index, data.batch

        if self.forward_on_top_features:
            x = torch.concat((data.x, data.top_features), dim = -1)
        else:
            x = data.x
        
        if self.gate_top_only:
            gate_features = data.top_features
        else:
            gate_features = x            
        
        for i, layer in enumerate(self.convs):
            if i == (self.depth - 1):
                x = F.dropout(x, p=0.5, training=self.training)

            if return_gate_res and gate_num_return == i: return layer(x, edge_index, gate_features, return_gate_res = True)
                
            x = layer(x, edge_index, gate_features)

        # Global mean pooling to create graph-level embeddings
        if not self.node_classification:
            x = global_mean_pool(x, batch)

        # Pass through the final fully connected layer
        x = self.fc(x)

        return x