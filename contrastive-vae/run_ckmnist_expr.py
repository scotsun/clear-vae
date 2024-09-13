import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import json
from expr.data import generate_style_dict, CKMNISTGenerator, CKMNIST
from corruption_utils import corruptions
from src.model import SimpleCNNClassifier, VAE
from src.trainer import SimpleCNNTrainer, CDVAETrainer, DownstreamMLPTrainer

TAU = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"

style_fns = [
    corruptions.identity,
    corruptions.stripe,
    corruptions.zigzag,
    corruptions.canny_edges,
    corruptions.spatter,
    corruptions.brightness,
]


def get_data_splits(k: int, seed: int):
    """
    Generate data splits and style dictionaries for CKMNIST dataset

    Parameters:
        k (int): The number of styles to generate for each class
        seed (int): The random seed for reproducibility

    Returns:
        style_dict (dict): A dictionary containing the style information for each class
        ckmnist_train (CKMNIST): The training set of the CKMNIST dataset
        ckmnist_valid (CKMNIST): The validation set of the CKMNIST dataset
        ckmnist_test (CKMNIST): The test set of the CKMNIST dataset
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    mnist = torchvision.datasets.MNIST("../data", train=True)
    mnist_train, mnist_test = random_split(mnist, [50000, 10000])
    style_dict = generate_style_dict(classes=list(range(10)), style_fns=style_fns, k=k)
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
    return style_dict, ckmnist_train, ckmnist_valid, ckmnist_test


def experiment(k, seed):
    style_dict, ckmnist_train, ckmnist_valid, ckmnist_test = get_data_splits(
        k=k, seed=seed
    )
    train_loader = DataLoader(ckmnist_train, batch_size=128, shuffle=True)
    valid_loader = DataLoader(ckmnist_valid, batch_size=128, shuffle=False)
    test_loader = DataLoader(ckmnist_test, batch_size=128, shuffle=False)

    # cnn pipeline
    cnn = SimpleCNNClassifier(n_class=10).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = SimpleCNNTrainer(
        cnn, optimizer, criterion, verbose_period=5, device=device
    )
    trainer.fit(epochs=41, train_loader=train_loader, valid_loader=valid_loader)
    (cnn_aupr_scores, cnn_auroc_scores), _ = trainer.evaluate(test_loader, False, 0)

    # vae+mlp pipeline
    vae = VAE(total_z_dim=16).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    trainer = CDVAETrainer(
        vae,
        optimizer,
        sim_fn="cosine",
        hyperparameter={
            "temperature": TAU,
            "beta": 1 / 4,
            "loc": 0,
            "scale": 1,
            "alpha": [50, 50],
            "label_flipping": True,
        },
        verbose_period=5,
        device=device,
    )
    trainer.fit(26, train_loader, valid_loader)
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
    trainer.fit(41, train_loader, valid_loader)
    (mlp_aupr_scores, mlp_auroc_scores), _ = trainer.evaluate(test_loader, False, 0)

    expr_output = dict()
    expr_output["cnn"] = {
        "pr": {
            "overall": np.mean(list(cnn_aupr_scores.values())).round(3),
            "stratified": cnn_aupr_scores,
        },
        "roc": {
            "overall": np.mean(list(cnn_auroc_scores.values())).round(3),
            "stratified": cnn_auroc_scores,
        },
    }
    expr_output["vae + mlp"] = {
        "pr": {
            "overall": np.mean(list(mlp_aupr_scores.values())).round(3),
            "stratified": mlp_aupr_scores,
        },
        "roc": {
            "overall": np.mean(list(mlp_auroc_scores.values())).round(3),
            "stratified": mlp_auroc_scores,
        },
    }

    print(expr_output)

    fpath = f"./expr_output/ckmnist/cmnist-k{k}-{seed}.json"
    with open(fpath, "w") as json_file:
        json.dump(expr_output, json_file, indent=4)


def main():
    for k in range(1, len(style_fns)):
        experiment(k=k, seed=int(np.random.randint(0, 1000)))


if __name__ == "__main__":
    main()
