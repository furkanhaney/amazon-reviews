import torch
import numpy as np
from model_lstm import NetSeq2Seq
import sentencepiece as spm

TOKEN_COUNT = 2048
WIDTH = 3
DEPTH = 3

net = NetSeq2Seq(TOKEN_COUNT, width=WIDTH, depth=DEPTH)
net.load_state_dict(torch.load("output/model.bin"))
net.eval()
print(net)

sp = spm.SentencePieceProcessor(model_file="processed/m.model")

last_token = 1
all_tokens = list(range(TOKEN_COUNT))
tokens = [1]

with torch.no_grad():
    while last_token != 3 and last_token != 2 and len(tokens) <= 100:
        x = torch.Tensor([tokens]).long()
        y = net(x).exp().numpy()
        last_token = np.random.choice(all_tokens, p=y[0, :, 0])
        tokens.append(last_token)
tokens = [int(token) for token in tokens]
print(tokens)
print(sp.decode(tokens))
