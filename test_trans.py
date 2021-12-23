import torch
import datasets
import pandas as pd
from transformers import (
    AlbertForSequenceClassification,
    AlbertTokenizerFast,
    Trainer,
    TrainingArguments,
)

model = AlbertForSequenceClassification.from_pretrained(
    "albert-base-v2", num_labels=5
).to(torch.device("cuda"))
model.train()
for param in model.albert.embeddings.parameters():
    param.requires_grad = False
print(model, '\n')
params_all = sum(p.numel() for p in model.parameters())
params_trainable = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)
print("Parameters: {:,}".format(params_all))
print("Trainable: {:,}".format(params_trainable), '\n')