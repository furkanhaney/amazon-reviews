import h5py
import torch
import pandas as pd
import numpy as np
import sentencepiece as spm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset


class ReviewClassification(Dataset):
    def __init__(self, split):
        self.split = split
        self.sp = spm.SentencePieceProcessor(model_file="processed/m.model")
        data_path = "processed/{}.csv".format(split)
        self.df = pd.read_csv(data_path)
        self.df["text"] = self.df["text"].map(str)
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.sp.encode(self.df.iloc[idx]["text"])
        y = self.df.iloc[idx]["label"]

        # Perform padding
        x = x[:256]
        x_np = np.ones((256,), dtype="int64") * 3
        x_np[: len(x)] = x
        return x_np, y


class ReviewModel(Dataset):
    def __init__(self, split):
        self.split = split
        self.sp = spm.SentencePieceProcessor(model_file="processed/m.model")
        data_path = "processed/{}.csv".format(split)
        self.df = pd.read_csv(data_path)
        self.df["text"] = self.df["text"].map(str)
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.sp.encode(self.df.iloc[idx]["text"])
        x = [self.sp.bos_id()] + x + [self.sp.eos_id()]
        y = x[1:]

        # Perform padding
        x = x[:256]
        x_np = np.ones((256,), dtype="int64") * 3
        x_np[: len(x)] = x

        y = y[:256]
        y_np = np.ones((256,), dtype="int64") * 3
        y_np[: len(y)] = y

        return x_np, y_np


def get_classification_loaders(
    splits=["train", "test"],
    in_memory=False,
    small=False,
    batch_size=32,
    num_workers=2,
    multiply=None,
):
    loaders = []
    for split in splits:
        dataset = ReviewClassification(split=split,)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        loaders.append(loader)
    return loaders


def get_lm_loaders(
    splits=["train", "test"],
    in_memory=False,
    small=False,
    batch_size=32,
    num_workers=2,
    multiply=None,
):
    loaders = []
    for split in splits:
        dataset = ReviewModel(split=split,)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            pin_memory=True,
            num_workers=num_workers,
        )
        loaders.append(loader)
    return loaders


if __name__ == "__main__":
    print(ReviewModel("test").__getitem__(0))
    # _, loader = get_loaders()
    # for batch_x, batch_y in loader:
    #     print(batch_x.shape)
    #     print(batch_y.shape)
    #     break
