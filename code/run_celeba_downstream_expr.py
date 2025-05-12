import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import json
from expr.expr_utils import kceleba_train_test_split

from src.utils.trainer_utils import (
    get_cnn_trainer,
    get_clearvae_trainer,
    get_cleartcvae_trainer,
    get_clearmimvae_trainer,
    get_hierarchical_vae_trainer,
)
from src.trainer import (
    DownstreamMLPTrainer,
    SimpleCNNTrainer,
    VAETrainer,
)
from src.utils.data_utils import get_process_celeba


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root_path", type=str, help="root path of the dataset")
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


def get_data(data_root_path):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    celeba = torchvision.datasets.CelebA(
        data_root_path,
        split="train",
        target_type="attr",
        transform=transform,
        download=True,
    )
    celeba = get_process_celeba(celeba)
    return celeba


def get_data_splits(celeba, k: int, seed: int):
    """
    Generate data splits and style dictionaries for k styled CelebA dataset
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    train, test, style_dict = kceleba_train_test_split(
        celeba_data=celeba, k=k, seed=seed
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


def experiment(celeba, k, seed, trainer_kwargs, epochs):
    print(f"Experiment: k={k}, seed={seed}")
    _, train, valid, test = get_data_splits(celeba, k=k, seed=seed)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=128, shuffle=False)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)

    models = {
        "baseline": (
            get_cnn_trainer,
            {
                "n_class": 10,
                "device": trainer_kwargs["device"],
                "cnn_arch": "SimpleCNN64Classifier",
                "in_channel": 3,
            },
        ),
        "gvae": (
            get_hierarchical_vae_trainer,
            {
                "beta": trainer_kwargs["beta"],
                "vae_lr": trainer_kwargs["vae_lr"],
                "z_dim": trainer_kwargs["z_dim"],
                "group_mode": "GVAE",
                "device": trainer_kwargs["device"],
                "vae_arch": trainer_kwargs["vae_arch"],
                "in_channel": 3,
            },
        ),
        "mlvae": (
            get_hierarchical_vae_trainer,
            {
                "beta": trainer_kwargs["beta"],
                "vae_lr": trainer_kwargs["vae_lr"],
                "z_dim": trainer_kwargs["z_dim"],
                "group_mode": "MLVAE",
                "device": trainer_kwargs["device"],
                "vae_arch": trainer_kwargs["vae_arch"],
                "in_channel": 3,
            },
        ),
        "clear": (
            get_clearvae_trainer,
            {"ps": True, **trainer_kwargs},
        ),
        "clear-tc": (
            get_cleartcvae_trainer,
            {"la": 1, "factor_cls_lr": 1e-4, **trainer_kwargs},
        ),
        "clear-mim (L1OutUB)": (
            get_clearmimvae_trainer,
            {
                "mi_estimator": "L1OutUB",
                "la": 3,
                "mi_estimator_lr": 2e-3,
                **trainer_kwargs,
            },
        ),
        "clear-mim (CLUB-S)": (
            get_clearmimvae_trainer,
            {
                "mi_estimator": "CLUBSample",
                "la": 3,
                "mi_estimator_lr": 2e-3,
                **trainer_kwargs,
            },
        ),
    }

    results = {}

    for model_name, (trainer_func, params) in models.items():
        print(f"\nTraining {model_name}:")
        trainer = trainer_func(**params)

        if isinstance(trainer, SimpleCNNTrainer):
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

    fpath = f"./expr_output/celeba/classification/celeba-k{k}-{seed}.json"
    with open(fpath, "w") as json_file:
        json.dump(results, json_file, indent=4)

    return results


def main():
    args = get_args()
    celeba = get_data(args.data_root_path)
    seed = int(np.random.randint(0, 1000))
    trainer_kwargs = {
        "beta": 1 / 32,
        "vae_arch": "VAE64",
        "vae_lr": 3e-5,
        "z_dim": 64,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "device": args.device,
        "in_channel": 3,
    }
    for k in range(1, 4):
        experiment(
            celeba,
            k=k,
            seed=seed,
            trainer_kwargs=trainer_kwargs,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
