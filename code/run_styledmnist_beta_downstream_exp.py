import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import json
from expr.expr_utils import generate_style_dict, KStyledMNISTGenerator, KStyledMNIST
from corruption_utils import corruptions

from src.utils.trainer_utils import (
    get_cnn_trainer,
    get_clearvae_trainer,
    get_cleartcvae_trainer,
    get_clearmimvae_trainer,
    get_hierarchical_vae_trainer,
)
from src.trainer import (
    DownstreamMLPTrainer,
    VAETrainer,
)


style_fns = [
    corruptions.identity,
    corruptions.stripe,
    corruptions.zigzag,
    corruptions.canny_edges,
    lambda x: corruptions.scale(x, 5),
    corruptions.brightness,
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs",
        type=int,
        default=41,
        help="num epochs; default 41",
    )
    parser.add_argument(
        "--alpha", type=float, default=1e2, help="penalty weight for snn; default 1e2"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature for snn; default 0.1",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run on; default cuda"
    )
    return parser.parse_args()


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
    style_dict = generate_style_dict(
        classes=list(range(10)), styles=list(range(len(style_fns))), k=k
    )
    mnist_generator = KStyledMNISTGenerator(mnist_train, style_fns, style_dict, "train")
    train = KStyledMNIST(
        mnist_generator,
        transforms.Compose([transforms.ToTensor(), lambda img: img / 255.0]),
    )
    mnist_generator = KStyledMNISTGenerator(mnist_test, style_fns, style_dict, "test")
    test = KStyledMNIST(
        mnist_generator,
        transforms.Compose([transforms.ToTensor(), lambda img: img / 255.0]),
    )
    train, valid = random_split(train, [0.85, 0.15])
    return style_dict, train, valid, test


def experiment_helper(
    train_loader,
    valid_loader,
    test_loader,
    vae_trainer: VAETrainer,
    epochs,
):
    # Train VAE
    vae_trainer.fit(epochs, train_loader, valid_loader)
    vae_trainer.model.eval()

    z_dim = vae_trainer.model.z_dim  # which is 1/2 * total_z_dim
    device = vae_trainer.device
    # Freeze VAE parameters
    for p in vae_trainer.model.parameters():
        p.requires_grad = False

    # Create and train MLP classifier
    mlp = nn.Sequential(
        nn.Linear(z_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    trainer = DownstreamMLPTrainer(
        vae_trainer.model, mlp, optimizer, criterion, 10, device
    )
    trainer.fit(epochs, train_loader, valid_loader)

    # Evaluate on test set
    (aupr_scores, auroc_scores), acc = trainer.evaluate(test_loader, False, 0)

    return aupr_scores, auroc_scores, acc


def experiment(beta, dataloaders, trainer_kwargs, epochs):
    print(f"Experiment: beta={beta}")

    train_loader, valid_loader, test_loader = dataloaders

    models = {
        "baseline": (
            get_cnn_trainer,
            {"n_class": 10, "device": trainer_kwargs["device"]},
        ),
        # "gvae": (
        #     get_hierachical_vae_trainer,
        #     {
        #         "beta": beta,
        #         "vae_lr": 5e-4,
        #         "z_dim": trainer_kwargs["z_dim"],
        #         "group_mode": "GVAE",
        #         "device": trainer_kwargs["device"],
        #     },
        # ),
        # "mlvae": (
        #     get_hierachical_vae_trainer,
        #     {
        #         "beta": beta,
        #         "vae_lr": 5e-4,
        #         "z_dim": trainer_kwargs["z_dim"],
        #         "group_mode": "MLVAE",
        #         "device": trainer_kwargs["device"],
        #     },
        # ),
        "clear": (
            get_clearvae_trainer,
            {"label_flipping": True, "beta": beta, **trainer_kwargs},
        ),
        # "clear-tc": (
        #     get_cleartcvae_trainer,
        #     {"la": 1, "factor_cls_lr": 1e-4, "beta": beta, **trainer_kwargs},
        # ),
        # "clear-mim (L1OutUB)": (
        #     get_clearmimvae_trainer,
        #     {
        #         "mi_estimator": "L1OutUB",
        #         "la": 3,
        #         "mi_estimator_lr": 2e-3,
        #         "beta": beta,
        #         **trainer_kwargs,
        #     },
        # ),
        # "clear-mim (CLUB-S)": (
        #     get_clearmimvae_trainer,
        #     {
        #         "mi_estimator": "CLUBSample",
        #         "la": 3,
        #         "mi_estimator_lr": 2e-3,
        #         "beta": beta,
        #         **trainer_kwargs,
        #     },
        # ),
    }

    results = {}

    for model_name, (trainer_func, params) in models.items():
        print(f"\nTraining {model_name}:")
        trainer = trainer_func(**params)

        if model_name == "baseline":
            trainer.fit(
                epochs=epochs, train_loader=train_loader, valid_loader=valid_loader
            )
            (aupr_scores, auroc_scores), acc = trainer.evaluate(test_loader, False, 0)
        else:
            aupr_scores, auroc_scores, acc = experiment_helper(
                train_loader, valid_loader, test_loader, trainer, epochs
            )

        results[model_name] = {
            "acc": round(float(acc), 3),
            "pr": {
                "overall": round(np.mean(list(aupr_scores.values())), 3),
                "stratified": aupr_scores,
            },
            "roc": {
                "overall": round(np.mean(list(auroc_scores.values())), 3),
                "stratified": auroc_scores,
            },
        }

    print("\nResults:")
    print(json.dumps(results, indent=2))

    fpath = f"./expr_output/styled-mnist/classification/styledmnist-beta{beta}.json"
    with open(fpath, "w") as json_file:
        json.dump(results, json_file, indent=4)

    return results


def main():
    args = get_args()
    np.random.seed(101)
    torch.manual_seed(101)
    _, train, valid, test = get_data_splits(k=3, seed=101)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=128, shuffle=False)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)

    trainer_kwargs = {
        "vae_lr": 5e-4,
        "z_dim": 16,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "device": args.device,
    }
    for b in [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2]:
        experiment(
            beta=b,
            dataloaders=(train_loader, valid_loader, test_loader),
            trainer_kwargs=trainer_kwargs,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
