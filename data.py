import torch
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset


def get_loaders(batch_size=32, num_workers=2, val_size=10000, test=False):
    x = torch.from_numpy(np.load("data/reviews.npy"))
    y = torch.from_numpy(np.load("data/ratings.npy")) - 1

    train_ds = TensorDataset(x[val_size:], y[val_size:])
    valid_ds = TensorDataset(x[:val_size], y[:val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    return train_loader, valid_loader, x.max().item() + 1, x.shape[1]
