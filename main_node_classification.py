import torch
import torch.optim as optim
import torch.nn as nn

from organ_data_with_top_features import get_Organ, generate_boolean_masks
from PlanetoidWithTopFeatures import PlanetoidWithTopFeatures
from train_utils_node_classification import train_model, test
from utils import get_stratified_split

import argparse

from CAMoE import CAMoE_GNN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default = 'CiteSeer')
parser.add_argument('--lr', type = float, default = 0.005)
parser.add_argument('--batch_size', type = int, default = 32)
parser.add_argument('--gnn_layer', type = str, default = 'GCN')
parser.add_argument('--max_epochs', type = int, default = 1000)
parser.add_argument('--patience', type = int, default = 100)
parser.add_argument('--num_folds', type = int, default = 5)
parser.add_argument('--split_idx', type = int, default = 0)
parser.add_argument('--hidden_dim', type = int, default = 32)
parser.add_argument('--depth', type = int, default = 3)
parser.add_argument('--heads', type = int, default = 1)
parser.add_argument('--GPU', type = bool, default = True)
parser.add_argument('--forward_on_top_features', type = bool, default = False)
parser.add_argument('--gate_on_top_only', type = bool, default = True)
parser.add_argument('--num_experts', type = int, default = 3)

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
EXPERTS = args.num_experts

if __name__ == "__main__":

    device = "cpu"
    if torch.cuda.is_available() and GPU:
        device = "cuda:0"

    print('====================')
    print(f'Device: {device}')
    print('====================\n')
    
    if DATA_SET_NAME == 'CiteSeer' or DATA_SET_NAME == 'Cora' or DATA_SET_NAME == 'PubMed': 
        dataset = PlanetoidWithTopFeatures(
            root = 'data/',
            name = DATA_SET_NAME)
        data = dataset[0]

    
    elif DATA_SET_NAME == 'Organ_C' or DATA_SET_NAME == 'Organ_S':
        data = get_Organ(view = DATA_SET_NAME[-1])
        
    else:
        raise Exception("Unsupported Dataset")

    print('====================')
    print(f'Dataset: {DATA_SET_NAME}')
    print('====================\n')

    train_indices, val_indices, test_indices = get_stratified_split(
        labels = data.y, 
        split_index = SPLIT_INDEX
        )
    
    train_mask, val_mask, test_mask = generate_boolean_masks(
        total_samples = len(data.y),
        train_indices = train_indices,
        val_indices = val_indices,
        test_indices = test_indices
        )

    model = CAMoE_GNN(
        node_features = data.num_features,
        hidden_dim = HIDDEN_DIMENSION,
        gnn_layer = GNN_LAYER,
        out_channels = len(torch.unique(data.y)),
        depth = DEPTH,
        heads = HEADS,
        forward_on_top_features = FORWARD_ON_TOP_FEATURES,
        gate_top_only = GATE_ON_TOP_ONLY,
        node_classification = True,
        experts = EXPERTS
        ).to(device)
    
    print('====================')
    print(f'Model:\n{model}')
    print('====================\n')


    optimizer = optim.Adam(model.parameters(), lr = LR)
    loss_fn = nn.CrossEntropyLoss()
    
    model, train_loss_arr, val_loss_arr, best_epoch, best_val_loss = train_model(
        model = model,
        optimizer = optimizer,
        loss_fn = loss_fn,
        data = data,
        train_mask = train_mask,
        val_mask = val_mask,
        max_epochs = MAX_EPOCHS,
        patience = PATIENCE,
        device = device,
        break_early = True,
        verbose = True
        )
    
    accuracy, precision, sensitivity, f1 = test(
        model = model,
        data = data, 
        mask = test_mask,
        device = device,
        verbose = True
        )
    
    del model
    del optimizer
    torch.cuda.empty_cache()

