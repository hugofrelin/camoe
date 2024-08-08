import json
import torch
from torch_geometric.data import Data, InMemoryDataset
import os
import hashlib

import networkx as nx
from torch_geometric.utils import to_networkx

def get_topological_node_features(data):
    num_top_features = 4

    G = to_networkx(data)
    
    # Compute betweenness centrality for each node
    betweenness_centrality = nx.betweenness_centrality(G)
    node_strength = {node: sum(weight for _, _, weight in G.edges(node, data='weight', default=1)) for node in G.nodes()}
    clustering_coefficient = nx.clustering(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100000)
    
    # Convert to tensor
    topology_tensor = torch.empty((data.num_nodes, num_top_features))

    topology_tensor[:, 0] = torch.tensor(list(node_strength.values()))
    topology_tensor[:, 1] = torch.tensor(list(betweenness_centrality.values()))
    topology_tensor[:, 2] = torch.tensor(list(clustering_coefficient.values()))
    topology_tensor[:, 3] = torch.tensor(list(eigenvector_centrality.values()))
    
    return topology_tensor

class PlanarSATPairsDatasetWithTopFeatures(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, compute_node_top_features = True):
        self.root = root
        self.compute_node_top_features = compute_node_top_features
        super(PlanarSATPairsDatasetWithTopFeatures, self).__init__(root, transform, pre_transform, pre_filter, compute_node_top_features)
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.verify_data()
        self.name = root.split('/')[-1]

    @property
    def raw_file_names(self):
        NAME = "GRAPHSAT"
        return [NAME + ".pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Load extracted attributes
        x_list = torch.load(os.path.join(self.root, 'x_list.pt'))
        edge_index_list = torch.load(os.path.join(self.root, 'edge_index_list.pt'))
        y_list = torch.load(os.path.join(self.root, 'y_list.pt'))

        # Create new data list
        data_list = []
        for i in range(len(x_list)):
            data = Data(x=x_list[i], edge_index=edge_index_list[i], y=y_list[i])
            data_list.append(data)

        # Apply filters and transforms if any
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if self.compute_node_top_features:
            for data in data_list:
                data.top_features = get_topological_node_features(data)

        # Collate data and save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def verify_data(self):
        # Load extracted attributes
        x_list = torch.load(os.path.join(self.root, 'x_list.pt'))
        edge_index_list = torch.load(os.path.join(self.root, 'edge_index_list.pt'))
        y_list = torch.load(os.path.join(self.root, 'y_list.pt'))

        # Function to calculate checksum
        def calculate_checksum(tensor_list):
            m = hashlib.sha256()
            for tensor in tensor_list:
                m.update(tensor.numpy().tobytes())
            return m.hexdigest()

        # Calculate checksums
        current_checksums = {
            'x_list': calculate_checksum(x_list),
            'edge_index_list': calculate_checksum(edge_index_list),
            'y_list': calculate_checksum(y_list),
        }

        # Load original checksums
        with open(os.path.join(self.root, 'checksums.json'), 'r') as f:
            original_checksums = json.load(f)

        # Compare checksums
        for key in original_checksums:
            if original_checksums[key] != current_checksums[key]:
                raise ValueError(f"Data verification failed for {key}")