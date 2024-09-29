import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import json
from expr.expr_utils import generate_style_dict, KStyledMNISTGenerator, KStyledMNIST
from corruption_utils import corruptions
from src.model import SimpleCNNClassifier, VAE
from src.trainer import (
    SimpleCNNTrainer,
    MLVAETrainer,
    CLEARVAETrainer,
    DownstreamMLPTrainer,
)

TAU = 0.1
SIM = "cosine"
EPOCHS = 41

device = "cuda" if torch.cuda.is_available() else "cpu"

style_fns = [
    corruptions.identity,
    corruptions.stripe,
    corruptions.zigzag,
    corruptions.canny_edges,
    lambda x: corruptions.scale(x, 5),
    corruptions.brightness,
]


def get_data_splits(k: int, seed: int):
    """
    Generate data splits and style dictionaries for k styled MNIST dataset

    Parameters:
        k (int): The number of styles to generate for each class
        seed (int): The random seed for reproducibility

    Returns:
        style_dict (dict): A dictionary containing the style information for each class
        train (k styled MNIST)
        valid (k styled MNIST)
        test (m-k styled MNIST)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    mnist = torchvision.datasets.MNIST("../data", train=True)
    mnist_train, mnist_test = random_split(mnist, [50000, 10000])
    style_dict = generate_style_dict(classes=list(range(10)), style_fns=style_fns, k=k)
    mnist_generator = KStyledMNISTGenerator(mnist_train, style_dict, "train")
    train = KStyledMNIST(
        mnist_generator,
        transforms.Compose([transforms.ToTensor(), lambda img: img / 255.0]),
    )
    mnist_generator = KStyledMNISTGenerator(mnist_test, style_dict, "test")
    test = KStyledMNIST(
        mnist_generator,
        transforms.Compose([transforms.ToTensor(), lambda img: img / 255.0]),
    )
    train_size = int(0.85 * len(train))
    train, valid = random_split(train, [train_size, len(train) - train_size])
    return style_dict, train, valid, test


def experiment(k, seed):
    style_dict, train, valid, test = get_data_splits(k=k, seed=seed)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=128, shuffle=False)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)

    # cnn pipeline
    print("baseline:")
    cnn = SimpleCNNClassifier(n_class=10).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = SimpleCNNTrainer(
        cnn, optimizer, criterion, verbose_period=5, device=device
    )
    trainer.fit(epochs=EPOCHS, train_loader=train_loader, valid_loader=valid_loader)
    (cnn_aupr_scores, cnn_auroc_scores), cnn_acc = trainer.evaluate(
        test_loader, False, 0
    )
    print()
    # ml-vae+mlp pipeline
    print("mlvae:")
    vae = VAE(total_z_dim=16).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    trainer = MLVAETrainer(
        vae,
        optimizer,
        hyperparameter={"beta": 1 / 8, "loc": 0, "scale": 1},
        verbose_period=5,
        device=device,
    )
    trainer.fit(EPOCHS, train_loader, valid_loader, with_evidence_acc=False)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    mlp = torch.nn.Sequential(
        torch.nn.Linear(8, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = DownstreamMLPTrainer(vae, mlp, optimizer, criterion, 10, device)
    trainer.fit(EPOCHS, train_loader, valid_loader)
    (mlvae_aupr_scores, mlvae_auroc_scores), mlvae_acc = trainer.evaluate(
        test_loader, False, 0
    )
    print()
    # clear-vae+mlp pipeline
    print("clear-vae:")
    vae = VAE(total_z_dim=16).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    trainer = CLEARVAETrainer(
        vae,
        optimizer,
        sim_fn=SIM,
        hyperparameter={
            "temperature": TAU,
            "beta": 1 / 8,
            "loc": 0,
            "scale": 1,
            "alpha": [1e2, 1e2],
            "label_flipping": True,
        },
        verbose_period=5,
        device=device,
    )
    trainer.fit(EPOCHS, train_loader, valid_loader)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    mlp = torch.nn.Sequential(
        torch.nn.Linear(8, 256),
        torch.nn.BatchNorm1d(256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    ).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = DownstreamMLPTrainer(vae, mlp, optimizer, criterion, 10, device)
    trainer.fit(EPOCHS, train_loader, valid_loader)
    (clearvae_aupr_scores, clearvae_auroc_scores), clearvae_acc = trainer.evaluate(
        test_loader, False, 0
    )
    print()
    expr_output = dict()
    expr_output["cnn"] = {
        "acc": round(float(cnn_acc), 3),
        "pr": {
            "overall": np.mean(list(cnn_aupr_scores.values())).round(3),
            "stratified": cnn_aupr_scores,
        },
        "roc": {
            "overall": np.mean(list(cnn_auroc_scores.values())).round(3),
            "stratified": cnn_auroc_scores,
        },
    }
    expr_output["ml-vae + mlp"] = {
        "acc": round(float(mlvae_acc), 3),
        "pr": {
            "overall": np.mean(list(mlvae_aupr_scores.values())).round(3),
            "stratified": mlvae_aupr_scores,
        },
        "roc": {
            "overall": np.mean(list(mlvae_auroc_scores.values())).round(3),
            "stratified": mlvae_auroc_scores,
        },
    }
    expr_output["clear-vae + mlp"] = {
        "acc": round(float(clearvae_acc), 3),
        "pr": {
            "overall": np.mean(list(clearvae_aupr_scores.values())).round(3),
            "stratified": clearvae_aupr_scores,
        },
        "roc": {
            "overall": np.mean(list(clearvae_auroc_scores.values())).round(3),
            "stratified": clearvae_auroc_scores,
        },
    }

    print(expr_output)

    fpath = f"./expr_output/cmnist/classification/cmnist-k{k}-{seed}.json"
    with open(fpath, "w") as json_file:
        json.dump(expr_output, json_file, indent=4)


def main():
    for k in range(1, len(style_fns)):
        experiment(k=k, seed=int(np.random.randint(0, 1000)))


if __name__ == "__main__":
    main()
