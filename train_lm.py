import argparse
from data2 import get_lm_loaders

# from model_cnn import Net
from model_lstm import NetSeq2Seq
from manta.training import ModelTrainer


if __name__ == "__main__":
    train_loader, valid_loader = get_lm_loaders(batch_size=128, num_workers=2)
    net = NetSeq2Seq(2048, width=3, depth=3)
    print(net, "\n")
    # print(classifier.encoder, '\n')

    trainer = ModelTrainer(
        net,
        train_loader,
        valid_loader,
        loss_fn="nll",
        metrics=["acc"],
        lr=1e-3,
        amp=True,
        gamma=0.5,
        step_size=1,
    )
    trainer.fit(epochs=5)
