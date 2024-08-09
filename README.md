# CAMoE: Centrality-Aware Mixture of Expers for Better Representation Learning in GNNs

This repository contains the implementation of the Centrality-Aware Mixture of Experts (CAMoE) framework, which integrates centrality measures into the Node Update process of Graph Neural Networks (GNNs) to enhance  adaptability and performance. Unlike traditional GNNs that apply uniform updates to all nodes, CAMoE utilizes toplogy-based gating and a Mixture of Experts (MoE) approach to enable adaptive Node Updates informed by node centrality.


CAMoE can be applied to any message passing GNN, and this implementation supports the Graph Convolutional Network (GCN)and the Graph Attention Network (GAT). The figure below illustrates the discrepancies between a standard GCN and a CAMoE-GCN.

![architecture](figures/architecture.png)

**a)** A flowchart illustrating how a graph described by the adjacency matrix $\mathbf{\hat{A}}$ and node feature matrix $\mathbf{H}^{(k)}$ is processed by a standard GCN (as described by [Kipf and Welling, 2016](https://arxiv.org/abs/1609.02907)) to generate $\mathbf{H}^{(k+1)}$. i) Shows the Neighbourhood Aggregation. ii) Displays the Node Update.

**b)** A flowchart showing how the same graph would be processed by the CAMoE-GCN. In addition to $\mathbf{\hat{A}}$ and $\mathbf{H}^{(k)}$, the graph is also described by a node centrality feature matrix $\mathbf{C}$. i) As this is the CAMoE-GCN, it uses the same aggregation method as the standard GCN. ii) The node centrality feature matrix is used to compute weights for each expert layer, with the SoftMax function applied for soft gating. iii) Each node is updated by the experts and then multiplied by the gating weights. iv) The sum of all weighted expert updates results in the new embedding $\mathbf{H}^{(k+1)}$.


