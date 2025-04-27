import argparse
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, ConcatDataset
import json
from pprint import pprint
from src.utils.trainer_utils import (
    get_cnn_trainer,
    get_lamcnn_trainer,
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


DOMAIN_CODE = {
    "art_painting": 0,
    "cartoon": 1,
    "photo": 2,
    "sketch": 3,
}


def _subsetting(pacs_dataset, content, style):
    subset = pacs_dataset.filter(
        lambda x: x["label"] == content and x["domain"] == style
    )
    return subset


def generate_style_dict(classes: list, styles: list, k: int):
    style_dict = {}
    for c in classes:
        train_styles = np.random.choice(styles, k, replace=False)
        test_styles = np.setdiff1d(styles, train_styles)
        style_dict[c] = {"train": train_styles.tolist(), "test": test_styles.tolist()}
    return style_dict


def get_data(data_root_path, classes: list, styles: list, k: int):
    pacs_dataset = load_from_disk(data_root_path)["train"]
    train_sets, test_sets = [], []
    style_dict = generate_style_dict(classes, styles, k)
    pprint(style_dict)
    for c in classes:
        for s in style_dict[c]["train"]:
            train_sets.append(_subsetting(pacs_dataset, c, s))
        for s in style_dict[c]["test"]:
            test_sets.append(_subsetting(pacs_dataset, c, s))
    train = ConcatDataset(train_sets)
    test = ConcatDataset(test_sets)
    train, valid = random_split(train, [0.85, 0.15])
    return generate_style_dict, train, valid, test


def collate_fn(batch):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    images = [transform(x["image"]) for x in batch]
    labels = [x["label"] for x in batch]
    domains = [DOMAIN_CODE[x["domain"]] for x in batch]
    return torch.stack(images), torch.tensor(labels), torch.tensor(domains)


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


def experiment(pacs_path, k, seed, trainer_kwargs, epochs):
    print(f"Experiment: k={k}, seed={seed}")
    _, train, valid, test = get_data(
        pacs_path, list(range(7)), ["art_painting", "cartoon", "photo", "sketch"], k
    )
    train_loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid, batch_size=128, shuffle=False, num_workers=4)
    test_loader = DataLoader(test, batch_size=128, shuffle=False, num_workers=4)

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
        "lam": (
            get_lamcnn_trainer,
            {
                "lam_coef": 0.1,
                "n_class": 10,
                "device": trainer_kwargs["device"],
                "cnn_arch": "LAMCNN64Classifier",
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
            {"label_flipping": True, **trainer_kwargs},
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

    fpath = fpath = f"./expr_output/pacs/classification/pacs-k{k}-{seed}.json"
    with open(fpath, "w") as json_file:
        json.dump(results, json_file, indent=4)

    return results


def main():
    args = get_args()
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
            pacs_path=args.data_root_path + "pacs",
            k=k,
            seed=seed,
            trainer_kwargs=trainer_kwargs,
            epochs=args.epochs,
        )


if __name__ == "__main__":
    main()
