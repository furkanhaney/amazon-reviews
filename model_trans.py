import torch
import torch.nn as nn
from transformers import (
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    XLNetForSequenceClassification,
    RobertaForSequenceClassification,
)
from transformers import (
    AlbertTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer)


def get_albert(v="albert-large-v2"):
    model = AlbertForSequenceClassification.from_pretrained(v, num_labels=5)
    tokenizer = AlbertTokenizer.from_pretrained(v)
    model_base = model.albert
    return model, tokenizer, model_base


def get_bert(v="bert-base-cased"):
    model = BertForSequenceClassification.from_pretrained(v, num_labels=5)
    tokenizer = BertTokenizer.from_pretrained(v)
    model_base = model.bert
    return model, tokenizer, model_base


def get_xlnet(v="xlnet-base-cased"):
    model = XLNetForSequenceClassification.from_pretrained(v, num_labels=5)
    tokenizer = XLNetTokenizer.from_pretrained(v)
    model_base = model.transformer
    return model, tokenizer, model_base

def get_roberta(v="roberta-large"):
    model = RobertaForSequenceClassification.from_pretrained(v, num_labels=5)
    tokenizer = RobertaTokenizer.from_pretrained(v)
    model_base = model.transformer
    return model, tokenizer, model_base


MODEL_INITIALIZERS = {
    "albert": get_albert,
    "bert": get_bert,
    "xlnet": get_xlnet,
    "roberta": get_roberta,
}


def get_model(model="albert", freeze_model=True, expand_classifier=False):
    model, tokenizer, model_base = MODEL_INITIALIZERS[model]()
    if expand_classifier:
        model.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 5),
        )
    model.train()
    if freeze_model:
        for param in model_base.parameters():
            param.requires_grad = False
    #     for param in model.bert.embeddings.parameters():
    #         param.requires_grad = False
    model = model.to(torch.device("cuda"))
    print(model, "\n")
    params_all = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Parameters: {:,}".format(params_all))
    print("Trainable: {:,}".format(params_trainable), "\n")
    return model, tokenizer, model_base
