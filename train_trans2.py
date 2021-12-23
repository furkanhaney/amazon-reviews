import torch
import datasets
import numpy as np
import pandas as pd
from data3 import get_loaders
from torch.optim import AdamW
from tqdm import tqdm
from manta.functions import compute_acc
from model_trans import get_model
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT = "output/albert_run/"
EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
ACCUMULATE = 1
SAVE_ITERS = 500

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def main():
    model, tokenizer, model_base = get_model(
        "bert",
        freeze_model=False,
        expand_classifier=False,
        # "albert", freeze_model=False, expand_classifier=False
    )
    train_loader, test_loader = get_loaders(
        tokenizer, batch_size=BATCH_SIZE, seq_length=512
    )
    opt = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        weight_decay=0,
    )
    criterion = torch.nn.CrossEntropyLoss()
    iters = 0
    avg_loss, avg_acc = 0, 0
    losses, accs = [], []

    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc="Epoch: {}/{}".format(epoch + 1, EPOCHS))
        with torch.cuda.amp.autocast():
            for i, batch in enumerate(pbar):
                input_ids, token_type_ids, attention_mask, labels = [
                    batch_t.cuda() for batch_t in batch
                ]
                y = model(input_ids, token_type_ids, attention_mask)
                loss = criterion(y["logits"], labels)
                loss.backward()
                if ACCUMULATE == 1 or i % ACCUMULATE == ACCUMULATE - 1:
                    opt.step()
                    opt.zero_grad()
                acc = compute_acc(y["logits"], labels)
                losses.append(loss.item())
                accs.append(acc.item())
                avg_loss = (avg_loss * i + loss.item()) / (i + 1)
                avg_acc = (avg_acc * i + acc.item()) / (i + 1)
                pbar.set_postfix_str(
                    "loss: {:.4f} acc: {:.2%}".format(avg_loss, avg_acc)
                )
                iters += 1
                if iters % SAVE_ITERS == 0:
                    df = pd.DataFrame(zip(losses, accs), columns=["loss", "acc"]).reset_index().rename(columns={"index": "iteration"})
                    df.to_csv(ROOT + "logs.csv", index=False)
                    ma = moving_average(df["acc"], 100)

                    plt.figure(dpi=300)
                    plt.title("Training Progress")
                    plt.grid(True)
                    plt.xlabel("Iteration")
                    plt.ylabel("Accuracy")
                    plt.ylim([0, 1])
                    plt.xlim([0, len(ma)])
                    plt.yticks(np.linspace(start=0, stop=1, num=11))
                    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
                    plt.plot(ma)
                    plt.savefig(ROOT + "acc.tiff")
                    torch.save(model.state_dict(), ROOT + "model.bin")


if __name__ == "__main__":
    main()
