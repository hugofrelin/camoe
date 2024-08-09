import numpy as np
import networkx as nx
import torch
import sklearn
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import os

def get_topological_node_features(data):
    print('Computing topological features')
    num_top_features = 4

    G = to_networkx(data)
    
    # Compute betweenness centrality for each node
    betweenness_centrality = nx.betweenness_centrality(G)
    print('betweenness_centrality: DONE')
    node_strength = {node: sum(weight for _, _, weight in G.edges(node, data='weight', default=1)) for node in G.nodes()}
    print('node_strength: DONE')
    clustering_coefficient = nx.clustering(G)
    print('clustering_coefficient: DONE')
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100000)
    print('eigenvector_centrality: DONE')
    
    # Convert to tensor
    topology_tensor = torch.empty((data.num_nodes, num_top_features))

    topology_tensor[:, 0] = torch.tensor(list(node_strength.values()))
    topology_tensor[:, 1] = torch.tensor(list(betweenness_centrality.values()))
    topology_tensor[:, 2] = torch.tensor(list(clustering_coefficient.values()))
    topology_tensor[:, 3] = torch.tensor(list(eigenvector_centrality.values()))
    
    return topology_tensor

def get_Organ(view = 'C', get_masks = False):    
    if view == 'C':
        dataset_name = 'organc'
    elif view == 'S':
        dataset_name = 'organs'
    
    data_label = np.load('organ_data/'+dataset_name+'/data_label.npy')

    data_feat = np.load('organ_data/'+dataset_name+'/data_feat.npy')
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(data_feat)
    data_feat = scaler.transform(data_feat)

    edge_index = np.load('organ_data/'+dataset_name+'/edge_index.npy')

    data_feat = torch.tensor(data_feat, dtype=torch.float)
    data_label = torch.tensor(data_label, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.int64)

    data = Data(x=data_feat, y=data_label, edge_index=edge_index)

    file_path = 'organ_data/'+dataset_name+'/topology_tensor.npy'
    if os.path.exists(file_path):
        topology_tensor = torch.tensor(np.load(file_path), dtype=torch.float)
    else:
        topology_tensor = get_topological_node_features(data)
        np.save(file_path, topology_tensor)

    data.top_features = topology_tensor
    
    return data 
    
def print_statistics(data, train_mask = None, val_mask = None, test_mask = None):
    
    print("=============== Dataset Properties ===============")
    print(f"Total Nodes: {data.x.size(0)}")
    print(f"Total Edges: {data.edge_index.size(1)}")
    print(f"Number of Features: {data.x.size(1)}")
    if len(data.y.size()) == 1:
        print(f"Number of Labels: {len(torch.unique(data.y))}")
        print("Task Type: Multi-class Classification")
    else:
        print(f"Number of Labels: {len(torch.unique(data.y))}")
        print("Task Type: Multi-label Classification")

    if train_mask != None and val_mask != None and test_mask != None:
        print(f"Training Nodes: {sum(train_mask)}")
        print(f"Validation Nodes: {sum(val_mask)}")
        print(f"Testing Nodes: {sum(test_mask)}")
    print()

def generate_boolean_masks(total_samples, train_indices, val_indices, test_indices):
    """
    Generate boolean masks for train, validation, and test splits based on provided indices.

    Args:
    - total_samples: Total number of samples in the dataset
    - train_indices: Indices for the training split
    - val_indices: Indices for the validation split
    - test_indices: Indices for the test split

    Returns:
    - train_mask: Boolean mask for the training split
    - val_mask: Boolean mask for the validation split
    - test_mask: Boolean mask for the test split
    """
    
    # Initialize all masks with False
    train_mask = torch.zeros(total_samples, dtype=torch.bool)
    val_mask = torch.zeros(total_samples, dtype=torch.bool)
    test_mask = torch.zeros(total_samples, dtype=torch.bool)
    
    # Set the indices to True for each split
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    return train_mask, val_mask, test_mask