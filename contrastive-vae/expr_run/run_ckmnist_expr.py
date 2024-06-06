import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

from expr.data import generate_style_dict, CKMNISTGenerator, CKMNIST
from corruption_utils import corruptions

from src.model import SimpleCNNClassifier, VAE
from src.trainer import SimpleCNNTrainer, CDVAETrainer, DownstreamMLPTrainer

device = "cuda" if torch.cuda.is_available() else "cpu"

K = 4

style_fns = [
    corruptions.identity,
    corruptions.stripe,
    corruptions.zigzag,
    corruptions.canny_edges,
    corruptions.spatter,
    corruptions.brightness,
]

torch.manual_seed(101)
mnist = torchvision.datasets.MNIST("../data", train=True)
mnist_train, mnist_test = random_split(mnist, [50000, 10000])


def main():
    style_dict = generate_style_dict(classes=list(range(10)), style_fns=style_fns, k=K)

    ckmnist_generator = CKMNISTGenerator(mnist_train, style_dict, "train")
    ckmnist_train = CKMNIST(
        ckmnist_generator,
        transforms.Compose([transforms.ToTensor(), lambda img: img / 255.0]),
    )

    ckmnist_generator = CKMNISTGenerator(mnist_test, style_dict, "test")
    ckmnist_test = CKMNIST(
        ckmnist_generator,
        transforms.Compose([transforms.ToTensor(), lambda img: img / 255.0]),
    )

    train_size = int(0.85 * len(ckmnist_train))
    ckmnist_train, ckmnist_valid = random_split(
        ckmnist_train, [train_size, len(ckmnist_train) - train_size]
    )

    train_loader = DataLoader(ckmnist_train, batch_size=128, shuffle=True)
    valid_loader = DataLoader(ckmnist_valid, batch_size=128, shuffle=False)
    test_loader = DataLoader(ckmnist_test, batch_size=128, shuffle=False)

    cnn = SimpleCNNClassifier(n_class=10).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = SimpleCNNTrainer(
        cnn, optimizer, criterion, verbose_period=5, device=device
    )
    trainer.fit(epochs=11, train_loader=train_loader, valid_loader=valid_loader)
