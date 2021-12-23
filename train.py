import argparse
from data import get_loaders
from model import Classifier
from manta.training import ModelTrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lr", help="Learning rate for the adam optimizer.", type=float, default=10e-4
    )
    parser.add_argument(
        "-g", "--gpu", help="Trains the model on gpu.", action="store_true"
    )
    parser.add_argument(
        "--epochs", help="Number of epochs to train.", type=int, default=10
    )
    parser.add_argument(
        "--batch_size", help="Size of minibatches.", type=int, default=128
    )
    parser.add_argument(
        "--num_workers",
        help="Number of workers for the data loaders.",
        type=int,
        default=2,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_loader, valid_loader, dict_size, review_length = get_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers
    )
    classifier = Classifier(review_length, dict_size)
    print(classifier)
    trainer = ModelTrainer(
        classifier, train_loader, valid_loader, metrics=["acc"], lr=args.lr
    )
    trainer.fit(epochs=args.epochs)
