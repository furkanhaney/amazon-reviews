import torch
import numpy as np
import torch.nn as nn

x = torch.from_numpy(np.load("data/reviews.npy"))
y = torch.from_numpy(np.load("data/ratings.npy"))
dict_size = x.max()
print(x.shape)
print("Dict Size", x.max())
embedding = nn.Embedding(dict_size, 64)

sample = x[1, :16]
print(sample)
print(embedding(sample))
