import torch.nn as nn
from manta.layers import GlobalMaxPooling
from manta.model import Model, Sequential

def get_conv(in_units, out_units):
    return Sequential(
        nn.Conv1d(in_units, out_units, kernel_size=3, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm1d(out_units),
    )

def get_block(in_units, out_units):
    return Sequential(
        get_conv(in_units, out_units),
        get_conv(out_units, out_units),
        nn.MaxPool1d(2),
    )


class Net(Model):
    def __init__(self, dict_length, width=1):
        super().__init__()
        width = 2 ** width
        self.embedding = nn.Embedding(dict_length, 300)
        self.encoder = Sequential(
            get_block(300, 16 * width),
            get_block(16 * width, 32 * width),
            get_block(32 * width, 32 * width),
            get_block(32 * width, 64 * width),
            get_block(64 * width, 64 * width),
            get_block(64 * width, 128 * width),
            get_block(128 * width, 128 * width),
            GlobalMaxPooling(),
        )
        self.lin = Sequential(
            nn.Linear(128 * width, 5),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        y = self.embedding(x)
        y = y.permute(0, 2, 1)
        y = self.encoder(y)
        y = self.lin(y)
        return y


if __name__ == "__main__":
    model = Net(50000, width=2)
    print(model)
