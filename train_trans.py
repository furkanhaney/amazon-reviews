import torch
import datasets
import pandas as pd
from transformers import AlbertForSequenceClassification
from data3 import get_loaders
from torch.optim import AdamW
from tqdm import tqdm
from manta.functions import compute_acc


def get_model():
    model = AlbertForSequenceClassification.from_pretrained(
        "albert-base-v2", num_labels=5
    ).to(torch.device("cuda"))
    model.train()
    # for param in model.albert.embeddings.parameters():
    #     param.requires_grad = False
    for param in model.albert.parameters():
        param.requires_grad = False
    print(model, '\n')
    params_all = sum(p.numel() for p in model.parameters())
    params_trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Parameters: {:,}".format(params_all))
    print("Trainable: {:,}".format(params_trainable), '\n')
    return model


def main():
    train_loader, test_loader = get_loaders(batch_size=4)
    model = get_model()
    opt = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=0)
    criterion = torch.nn.CrossEntropyLoss()
    
    avg_loss = 0
    avg_acc = 0
    pbar = tqdm(train_loader)
    with torch.cuda.amp.autocast():
        for i, batch in enumerate(pbar):
            if i == 2000:
                for param in model.albert.parameters():
                    param.requires_grad = True
                for g in opt.param_groups:
                    g['lr'] = 1e-6
            input_ids, token_type_ids, attention_mask, labels = [batch_t.cuda() for batch_t in batch] 

            opt.zero_grad()
            y = model(input_ids, token_type_ids, attention_mask)
            loss = criterion(y["logits"], labels)
            loss.backward()
            opt.step()
            acc = compute_acc(y["logits"], labels)

            avg_loss = (avg_loss * i + loss.item()) / (i + 1)
            avg_acc = (avg_acc * i + acc.item()) / (i + 1)
            pbar.set_postfix_str("loss: {:.4f} acc: {:.2%}".format(avg_loss, avg_acc))

if __name__ == "__main__":
    main()
