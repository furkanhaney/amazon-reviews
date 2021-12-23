import torch.nn as nn
from manta.layers.layers import GlobalAvgPooling
from manta.layers.conv import Conv1D


class Classifier(nn.Module):
    def __init__(self, review_length, dict_length):
        super().__init__()
        self.embedding = nn.Embedding(dict_length, 64)
        self.seq = nn.Sequential(
            Conv1D(64, 64),
            nn.AvgPool1d(2),  # 256
            Conv1D(64, 128),
            nn.AvgPool1d(2),  # 128
            Conv1D(128, 128),
            nn.AvgPool1d(2),  # 64
            Conv1D(128, 256),
            nn.AvgPool1d(2),  # 32
            Conv1D(256, 256),
            nn.AvgPool1d(2),  # 16
            Conv1D(256, 512),
            nn.AvgPool1d(2),  # 8
            Conv1D(512, 512),
            GlobalAvgPooling(),
            nn.Linear(512, 5),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        y = self.embedding(x)
        y = y.permute(0, 2, 1)
        y = self.seq(y)
        return y

    def __str__(self):
        num_params = sum(p.numel() for p in self.seq.parameters())
        return super().__str__() + "\nTotal Parameters: {:,}\n".format(num_params)


if __name__ == "__main__":
    model = Classifier(512, 50000)
    print(model)
