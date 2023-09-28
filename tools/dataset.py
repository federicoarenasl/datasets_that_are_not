
import torch
import numpy as np
from torchvision.transforms.transforms import ToTensor
from typing import List, Tuple

def loader_from_indices(
    dataset: torch.utils.data.Dataset,
    indices: List[int],
    batch_size: int = 1
) -> torch.utils.data.DataLoader:
    """
    Create a dataloader from a dataset and a list of indices.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to use
    indices : List[int]
        List of indices to use
    """
    sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
    return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size)


def split_dataset(
    dataset: torch.utils.data.Dataset,
    validation_split: float = 0.2,
    random_seed: int = 42,
    batch_size: int = 1
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Load the training and validation splits of a dataset into dataloaders.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to split
    validation_split : float
        Percentage of the dataset to use for validation
    random_seed : int
        Random seed to use for splitting the dataset
    batch_size : int
        Batch size to use for the dataloaders
    
    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
        Training and validation dataloaders.
    """
    # Split dataset into training and validation sets
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(validation_split * num_train))
    # Shuffle indices
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    # Create dataloaders
    train_dataloader = loader_from_indices(dataset, train_idx, batch_size=batch_size) 
    validation_dataloader = loader_from_indices(dataset, valid_idx, batch_size=batch_size)

    return train_dataloader, validation_dataloader
