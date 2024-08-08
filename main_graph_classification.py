import torch
import torch.optim as optim
import torch.nn as nn


from TUDatasetWithTopFeatures import TUDatasetWithTopFeatures
from PlanarSATPairsDatasetWithTopFeatures import PlanarSATPairsDatasetWithTopFeatures
from train_utils_graph_classification import *

import argparse

from CAMoE import CAMoE_GNN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MUTAG')
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gnn_layer', type=str, default='GCN')
parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num_folds', type=int, default=5)
parser.add_argument('--split_idx', type=int, default=0)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--depth', type=int, default=2)
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--GPU', type=bool, default=True)
parser.add_argument('--forward_on_top_features', type=bool, default=False)
parser.add_argument('--gate_on_top_only', type=bool, default=True)

args = parser.parse_args()

DATA_SET_NAME = args.dataset
GNN_LAYER = args.gnn_layer
MAX_EPOCHS = args.max_epochs
PATIENCE = args.patience
NUM_FOLDS = args.num_folds
SPLIT_INDEX = args.split_idx
HIDDEN_DIMENSION = args.hidden_dim
DEPTH = args.depth
HEADS = args.heads
LR = args.lr
BATCH_SIZE = args.batch_size
GPU = args.GPU
FORWARD_ON_TOP_FEATURES = args.forward_on_top_features
GATE_ON_TOP_ONLY = args.gate_on_top_only

if __name__ == "__main__":

    device = "cpu"
    if torch.cuda.is_available() and GPU:
        device = "cuda:0"

    print(f'Device: {device}')
    
    if DATA_SET_NAME == 'MUTAG' or DATA_SET_NAME == 'PROTEINS': 
        dataset = TUDatasetWithTopFeatures(
            root='data',
            name=DATA_SET_NAME,
            )
    
    elif DATA_SET_NAME == 'EXP' or DATA_SET_NAME == 'CEXP':
        dataset = PlanarSATPairsDatasetWithTopFeatures(
            root="expData/" + DATA_SET_NAME)
        
    else:
        raise Exception("Unsupported Dataset")

    print(f'Dataset: {DATA_SET_NAME}')

    train_indices, val_indices, test_indices = get_stratified_split(
        labels = dataset.y,
        split_index = SPLIT_INDEX,
        k = NUM_FOLDS
        )
    
    train_loader = DataLoader(dataset[train_indices], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset[val_indices], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset[test_indices], batch_size=BATCH_SIZE, shuffle=True)
    
    model = CAMoE_GNN(
        node_features=dataset.num_features,
        hidden_dim=HIDDEN_DIMENSION,
        gnn_layer=GNN_LAYER,
        out_channels=dataset.num_classes,
        depth=DEPTH,
        heads=HEADS,
        forward_on_top_features = FORWARD_ON_TOP_FEATURES,
        gate_top_only = GATE_ON_TOP_ONLY,
        node_classification = False
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    model, train_loss_arr, val_loss_arr, best_epoch, best_val_loss = train_model(
        model = model,
        optimizer = optimizer,
        loss_fn = loss_fn,
        train_loader = train_loader,
        val_loader = val_loader,
        verbose = True,
        max_epochs = MAX_EPOCHS,
        patience = PATIENCE,
        device = device,
        break_early = True
        )
    
    accuracy, precision, sensitivity, f1 = test(
        model = model,
        data_loader = test_loader,
        device=device,
        verbose=True
        )
    
    del model
    del optimizer
    torch.cuda.empty_cache()  # Clear CUDA memory

