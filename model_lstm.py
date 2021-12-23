import torch
import torch.nn as nn
import torch.nn.utils
from manta.model import Model, Sequential


class Net(Model):
    def __init__(self, dict_size, embedding_size=300, width=1, depth=2, dropout=0):
        super().__init__()
        width = 2 ** width
        self.embedding = nn.Embedding(dict_size, embedding_size)
        self.encoder = Sequential(
            nn.LSTM(embedding_size, 32 * width, bidirectional=True, num_layers=depth, dropout=dropout),
        )
        self.lin = Sequential(nn.Linear(64 * width, 5), nn.LogSoftmax(dim=1),)

    def forward(self, x):
        y = self.embedding(x)
        y, _ = self.encoder(y)
        y = torch.max(y, dim=1)[0]
        y = self.lin(y)
        return y


class NetSeq2Seq(Model):
    def __init__(self, dict_size, embedding_size=300, width=1, depth=2):
        super().__init__()
        width = 2 ** width
        self.embedding = nn.Embedding(dict_size, embedding_size)
        self.encoder = []
        self.norms = []
        current_dim = embedding_size
        for i in range(depth):
            next_dim = 32 * width
            self.encoder.append(nn.LSTM(current_dim, next_dim))
            self.norms.append(nn.LayerNorm(next_dim))
            # self.norms.append(nn.Dropout(0.2))
            current_dim = next_dim
        self.encoder = nn.Sequential(*self.encoder)
        self.norms = nn.Sequential(*self.norms)
        self.lin = Sequential(nn.Linear(32 * width, dict_size), nn.LogSoftmax(dim=2),)

    def forward(self, x):
        y = self.embedding(x)
        for layer, norm in zip(self.encoder, self.norms):
            y, _ = layer(y)
            y = norm(y)
        y = self.lin(y)
        return y.permute(0, 2, 1)


if __name__ == "__main__":
    # net = Net(width=2)
    # print(net, '\n')
    # x = torch.zeros((8, 512, 768))
    # y = net(x)
    # print(y.shape)
    net = NetSeq2Seq(dict_size=8000, width=2)
    print(net, "\n")
    x = torch.zeros((8, 512), dtype=torch.long)
    y = net(x)
    print(x.shape)
    print(y.shape)
