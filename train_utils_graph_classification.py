from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, Callable, List

def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Counts the total and trainable parameters of the model.

    Args:
        model (nn.Module): The neural network model.

    Returns:
        Tuple[int, int]: A tuple containing the total number of parameters and the number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer, loss_fn: Callable, device: str = 'cuda') -> float:
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for providing the data.
        optimizer (optim.Optimizer): Optimizer to update model parameters.
        loss_fn (Callable): Loss function used to compute the loss.
        device (str, optional): Device to run the training on ('cuda' or 'cpu').

    Returns:
        float: Average loss for this training epoch.
    """
    model.train()  
    total_loss = 0
    count = 0

    for batch in data_loader:
        batch = batch.to(device)  
        optimizer.zero_grad()  
        out = model(batch)  

        loss = loss_fn(out, batch.y)
        loss.backward()  
        optimizer.step() 
        
        total_loss += loss.item() * batch.num_graphs 
        count += batch.num_graphs

    average_loss = total_loss / count if count > 0 else 0
    return average_loss

def val_epoch(model: nn.Module, data_loader: DataLoader, loss_fn: Callable, device: str = 'cuda') -> float:
    """
    Validates the model for one epoch.

    Args:
        model (nn.Module): The neural network model to validate.
        data_loader (DataLoader): DataLoader for providing the validation data.
        loss_fn (Callable): Loss function used to compute the loss.
        device (str, optional): Device to run the validation on ('cuda' or 'cpu').

    Returns:
        float: Average loss for this validation epoch.
    """
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch)

            loss = loss_fn(out, batch.y)

            total_loss += loss.item() * batch.num_graphs
            count += batch.num_graphs

    average_loss = total_loss / count if count > 0 else 0
    return average_loss

def test(model: nn.Module, data_loader: DataLoader, device: str = 'cuda', verbose: bool = True) -> Tuple[float, float, float, float]:
    """
    Tests the model using the provided DataLoader and computes performance metrics.

    Args:
        model (nn.Module): The neural network model to evaluate.
        data_loader (DataLoader): DataLoader providing the test data.
        device (str): The computing device ('cuda' or 'cpu').
        verbose (bool): If True, print the metrics after testing.

    Returns:
        Tuple[float, float, float, float]: A tuple containing accuracy, precision, sensitivity (recall), and F1-score.
    """
    model.eval()
    model.to(device)
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            predicted = output.argmax(dim=1)  

            true_labels.extend(data.y.cpu().tolist())
            predicted_labels.extend(predicted.cpu().tolist())

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    sensitivity = recall_score(true_labels, predicted_labels, average='macro')
    f1 = f1_score(true_labels, predicted_labels, average='macro')

    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"F1-score: {f1:.4f}")

    return accuracy, precision, sensitivity, f1

def train_model(model: nn.Module, optimizer: optim.Optimizer, loss_fn: Callable, train_loader: DataLoader, val_loader: DataLoader, 
                max_epochs: int = 1000, patience: int = 50, device: str = 'cuda', verbose: bool = True, break_early: bool = True) -> Tuple[nn.Module, List[float], List[float], int, float]:
    """
    Trains the model with early stopping based on validation loss.

    Args:
        model (nn.Module): The neural network model to train.
        optimizer (optim.Optimizer): Optimizer to update model parameters.
        loss_fn (Callable): Loss function used to compute the loss.
        train_loader (DataLoader): DataLoader for providing the training data.
        val_loader (DataLoader): DataLoader for providing the validation data.
        max_epochs (int, optional): Maximum number of epochs to train. Default is 1000.
        patience (int, optional): Number of epochs to wait for improvement before stopping early. Default is 50.
        device (str, optional): Device to run the training on ('cuda' or 'cpu'). Default is 'cuda'.
        verbose (bool, optional): If True, print training progress. Default is True.
        break_early (bool, optional): If True, break training early if no improvement. Default is True.

    Returns:
        Tuple[nn.Module, List[float], List[float], int, float]: The trained model, training loss history, validation loss history, best epoch, and best validation loss.
    """
    best_val_loss = float('inf')
    improvement_search = True
    not_improved_counter = 0
    best_epoch = 0
    best_model_state = None

    train_loss_arr = []
    val_loss_arr = []

    for epoch in range(max_epochs):

        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device=device)

        val_loss = val_epoch(model, val_loader, loss_fn, device=device)

        train_loss_arr.append(train_loss)
        val_loss_arr.append(val_loss)

        if improvement_search:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                not_improved_counter = 0
                best_model_state = model.state_dict()

            else:
                not_improved_counter += 1

            if verbose and (epoch+1)%20 == 0:
                print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss}")
            
            if not_improved_counter >= patience:
                improvement_search = False
                if break_early:
                    break

    model.load_state_dict(best_model_state)

    return model, train_loss_arr, val_loss_arr, best_epoch, best_val_loss

def get_stratified_split(labels: torch.Tensor, split_index: int, k: int = 5, random_state: int = 42) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns the indices for the specified stratified split.

    Args:
    labels (torch.Tensor): A tensor of labels.
    k (int): The number of folds.
    split_index (int): The index of the split to retrieve (0-based).

    Returns:
    tuple of torch.Tensor: Train indices, validation indices, and test indices.
    """
    labels_np = labels.numpy()
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    folds = []

    for _, test_index in skf.split(torch.zeros(len(labels_np)), labels_np):
        folds.append(test_index)

    if split_index < 0 or split_index >= k:
        raise ValueError("split_index must be between 0 and k-1")

    test_indices = torch.tensor(folds[split_index])
    val_indices = torch.tensor(folds[(split_index + 1) % k])
    train_indices = torch.cat([torch.tensor(folds[i]) for i in range(k) if i != split_index and i != (split_index + 1) % k])

    assert torch.any(torch.isin(train_indices, val_indices)).item() == False
    assert torch.any(torch.isin(train_indices, test_indices)).item() == False
    assert torch.any(torch.isin(test_indices, val_indices)).item() == False

    return train_indices, val_indices, test_indices