import h5py
import torch
import pandas as pd
import numpy as np
import sentencepiece as spm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from transformers import AlbertTokenizer
from transformers import AlbertForSequenceClassification


class ReviewDataset(Dataset):
    def __init__(self, tokenizer, split, seq_length=512):
        self.split = split
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        data_path = "processed/{}.csv".format(split)
        self.df = pd.read_csv(data_path)
        self.df["text"] = self.df["text"].map(str)
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = self.tokenizer(
            self.df.iloc[idx]["text"],
            padding="max_length",
            max_length=self.seq_length,
            truncation=True,
        )
        y = self.df.iloc[idx]["label"]
        return x, y


def custom_collate(samples):
    input_ids = torch.tensor([sample[0]["input_ids"] for sample in samples])
    token_type_ids = torch.tensor([sample[0]["token_type_ids"] for sample in samples])
    attention_mask = torch.tensor([sample[0]["attention_mask"] for sample in samples])
    labels = torch.tensor([sample[1] for sample in samples])
    return input_ids, token_type_ids, attention_mask, labels


def get_loaders(
    tokenizer,
    splits=["train", "test"],
    in_memory=False,
    small=False,
    batch_size=32,
    num_workers=2,
    multiply=None,
    seq_length=512,
):
    loaders = []
    for split in splits:
        dataset = ReviewDataset(tokenizer=tokenizer, split=split, seq_length=seq_length)
        loader = DataLoader(
            dataset,
            collate_fn=custom_collate,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=num_workers,
        )
        loaders.append(loader)
    return loaders


if __name__ == "__main__":
    pass
    # _, loader = get_loaders(batch_size=32, num_workers=0)
    # model = AlbertForSequenceClassification.from_pretrained(
    #     "albert-base-v2", num_labels=5
    # )
    # for input_ids, token_type_ids, attention_mask, labels in loader:
    #     y = model(input_ids, token_type_ids, attention_mask)
    #     print(y)
    #     break
