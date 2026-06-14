"""Data loading and partitioning utilities for federated learning."""

import os
import pathlib

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split


def generate_distributed_datasets(k, alpha, save_dir):
    """Partition FashionMNIST across k clients using a Dirichlet distribution.

    Higher alpha = more uniform distribution across classes (closer to IID).
    Lower alpha = more skewed (non-IID), each client sees fewer classes.
    """
    # Download and load the full FashionMNIST training set
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.FashionMNIST(
        root="./data_raw", train=True, download=True, transform=transform
    )

    targets = np.array(dataset.targets)  # class labels for every sample
    num_classes = 10

    # Collect indices for each class
    indices_per_class = [np.where(targets == c)[0] for c in range(num_classes)]

    # client_indices[i] will hold the list of dataset indices for client i
    client_indices = [[] for _ in range(k)]

    for class_indices in indices_per_class:
        # Sample how much of this class each client should get
        proportions = np.random.dirichlet([alpha] * k)
        np.random.shuffle(class_indices)

        # Convert proportions to integer counts (fix rounding on last client)
        counts = (proportions * len(class_indices)).astype(int)
        counts[-1] = len(class_indices) - counts[:-1].sum()

        start = 0
        for cid, count in enumerate(counts):
            client_indices[cid].extend(class_indices[start : start + count].tolist())
            start += count

    # Save each client's subset as a .pt file
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for cid, idxs in enumerate(client_indices):
        # Stack all images and labels for this client
        x = torch.stack([dataset[i][0] for i in idxs])
        y = torch.tensor([dataset[i][1] for i in idxs])
        torch.save((x, y), save_path / f"client_{cid}.pt")
        print(f"Client {cid}: {len(idxs)} samples")


def load_client_data(cid, data_dir, batch_size):
    """Load a single client's dataset and return train/val DataLoaders.

    Splits the client's data 80% train / 20% validation.
    """
    path = pathlib.Path(data_dir) / f"client_{cid}.pt"
    x, y = torch.load(path, weights_only=True)

    dataset = TensorDataset(x, y)

    # 80/20 train-val split
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
