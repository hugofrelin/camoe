import torch
from sklearn.model_selection import StratifiedKFold
from typing import Tuple

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