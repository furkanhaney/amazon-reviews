import numpy as np
import pandas as pd
from tqdm import tqdm
import sentencepiece as spm
import matplotlib.pyplot as plt
import multiprocessing as mp

sp = spm.SentencePieceProcessor(model_file="processed/m.model")


def batchify(data, batch_size):
    batch_count = -(-len(data) // batch_size)
    for i in range(batch_count):
        yield data[i * batch_size : (i + 1) * batch_size]


def batch_encode(batch):
    return [len(sp.encode(x)) for x in batch]


def main():
    df = pd.read_csv("processed/train.csv")

    pool = mp.Pool(8)
    counts = []
    batches = list(batchify(df["review_body"].map(str).tolist(), batch_size=4096))
    pbar = tqdm(pool.imap_unordered(batch_encode, batches), total=len(batches))
    for batch_counts in pbar:
        counts += batch_counts

    counts = np.array(counts)
    counts = counts[counts <= 1024]
    for i in range(7):
        length = 2 ** (i + 4)
        ratio = (counts < length).mean()
        print("Tokens: {} Ratio: {:.2%}".format(length, ratio))

    plt.figure(figsize=(8, 8))
    plt.hist(counts, bins=100)
    plt.show()


if __name__ == "__main__":
    main()
