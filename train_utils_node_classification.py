import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params

def train_epoch(model, data, train_mask, optimizer, loss_fn, device='cuda'):
    """
    Trains the model for one epoch.

    Parameters:
        model (torch.nn.Module): The neural network model to train.
        data_loader (torch.utils.data.DataLoader): DataLoader for providing the data.
        optimizer (torch.optim.Optimizer): Optimizer to update model parameters.
        loss_fn (callable): Loss function used to compute the loss.
        device (str, optional): Device to run the training on ('cuda' or 'cpu').

    Returns:
        float: Average loss for this training epoch.
    """
    model.train()

    data = data.to(device)
    optimizer.zero_grad()  # Clear previous gradients
    out = model(data)  # Get model predictions

    loss = loss_fn(out[train_mask], data.y[train_mask])
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update the model parameters

    return loss.item()

def val_epoch(model, data, val_mask, loss_fn, device='cuda'):
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        out = model(data)
        loss = loss_fn(out[val_mask], data.y[val_mask])
    return loss.item()

def test(model, data, mask, device='cuda', verbose=True):
    model.eval()
    data = data.to(device)

    true_labels = data.y.cpu()

    with torch.no_grad():
        predicted_labels = model(data).argmax(1).cpu()

    accuracy = accuracy_score(true_labels[mask], predicted_labels[mask])
    precision = precision_score(true_labels[mask], predicted_labels[mask], average='macro')
    sensitivity = recall_score(true_labels[mask], predicted_labels[mask], average='macro')
    f1 = f1_score(true_labels[mask], predicted_labels[mask], average='macro')

    if verbose:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"F1-score: {f1:.4f}")
    
    return accuracy, precision, sensitivity, f1

def train_model(model, optimizer, loss_fn, data, train_mask, val_mask, max_epochs = 1000, patience = 50, device='cuda', verbose=True, break_early = True):
    best_val_loss = float('inf')
    improvement_search = True
    not_improved_counter = 0
    best_epoch = 0
    best_model_state = None

    train_loss_arr = []
    val_loss_arr = []

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, data, train_mask, optimizer, loss_fn, device)
        val_loss = val_epoch(model, data, val_mask, loss_fn, device)

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
            
            # Check early stopping condition
            if not_improved_counter >= patience:
                improvement_search = False
                if break_early:
                    break
    
    model.load_state_dict(best_model_state)

    return model, train_loss_arr, val_loss_arr, best_epoch, best_val_loss