import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from src.model import VAE
from src.trainer import CLEARVAETrainer, HierachicalVAETrainer, SEPVAETrainer
from src.utils import StyledMNISTGenerator, StyledMNIST
from corruption_utils import corruptions

CORRUPTION_FNS = {
    corruptions.identity: 0.15,
    corruptions.stripe: 0.2,
    corruptions.zigzag: 0.25,
    corruptions.canny_edges: 0.1,
    lambda x: corruptions.scale(x, 5): 0.1,
    corruptions.brightness: 0.2,
}
# BETAS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]
BETAS = [1]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=101, help="random seed; default 101"
    )
    parser.add_argument(
        "--alpha", type=float, default=1e2, help="penalty weight for snn; default 1e2"
    )
    parser.add_argument("--epochs", type=int, default=16, help="num epochs")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature for snn; default 0.1",
    )
    parser.add_argument(
        "--z_dim", type=int, default=16, help="latent dimension; default 16"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run on; default cuda"
    )
    return parser.parse_args()


def get_data(seed):
    # set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # generate data
    mnist = torchvision.datasets.MNIST("../data", train=True, download=True)
    generator = StyledMNISTGenerator(mnist, CORRUPTION_FNS)
    dataset = StyledMNIST(
        generator, transforms.Compose([transforms.ToTensor(), lambda img: img / 255.0])
    )
    train, valid, test = random_split(dataset, [40_000, 10_000, 10_000])
    return train, valid, test


def get_hierachical_vae_trainer(beta, z_dim, group_mode, device):
    vae = VAE(total_z_dim=z_dim, group_mode=group_mode).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)
    trainer = HierachicalVAETrainer(
        vae,
        optimizer,
        hyperparameter={
            "beta": beta,
            "scale": 1,
            "loc": 0,
        },
        verbose_period=5,
        device=device,
    )
    return trainer


def get_clearvae_trainer(beta, label_flipping, z_dim, alpha, temperature, device):
    vae = VAE(total_z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)
    trainer = CLEARVAETrainer(
        vae,
        optimizer,
        sim_fn="cosine",
        hyperparameter={
            "temperature": temperature,
            "alpha": [alpha, alpha],
            "beta": beta,
            "label_flipping": label_flipping,
            "loc": 0,
            "scale": 1,
        },
        verbose_period=5,
        device=device,
    )
    return trainer


def get_sepvae_trainer(beta, la, z_dim, alpha, temperature, device):
    vae = VAE(total_z_dim=z_dim).to(device)
    factor_cls = nn.Sequential(
        nn.Linear(z_dim, z_dim),
        nn.ReLU(),
        nn.Linear(z_dim, 1),
        nn.Sigmoid(),
    ).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)
    factor_optimizer = torch.optim.Adam(factor_cls.parameters(), lr=1e-4)
    trainer = SEPVAETrainer(
        vae,
        factor_cls,
        optimizers={"vae_optim": vae_optimizer, "factor_optim": factor_optimizer},
        sim_fn="cosine",
        hyperparameter={
            "temperature": temperature,
            "alpha": alpha,
            "beta": beta,
            "loc": 0,
            "scale": 1,
            "lambda": la,
        },
        verbose_period=5,
        device=device,
    )
    return trainer


class ExperimentHelper:
    def __init__(self):
        self.mig_list = []
        self.elbo_list = []

    def run_experiment(
        self,
        trainer,
        train_loader,
        valid_loader,
        test_loader,
        epochs,
    ):
        print(f"----------{type(trainer)}------------")

        if isinstance(trainer, HierachicalVAETrainer):
            trainer.fit(epochs, train_loader, valid_loader, eval_evidence_acc=True)
            mig, elbo = trainer.evaluate(test_loader, False, 0, with_evidence_acc=False)
        else:
            trainer.fit(epochs, train_loader, valid_loader)
            mig, elbo = trainer.evaluate(test_loader, False, 0)

        print("")
        self.mig_list.append(mig)
        self.elbo_list.append(elbo)

    def get_results(self):
        return self.mig_list, self.elbo_list


def main():
    args = get_args()
    train, valid, test = get_data(args.seed)
    loaders = {
        "train": DataLoader(train, batch_size=128, shuffle=True),
        "valid": DataLoader(valid, batch_size=128, shuffle=False),
        "test": DataLoader(test, batch_size=128, shuffle=False),
    }

    default_hyperparam_kwargs = {
        "z_dim": args.z_dim,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "device": args.device,
    }
    trainer_kwargs = {**loaders, "epochs": args.epochs}

    models = {
        "ps": lambda beta: get_clearvae_trainer(
            beta=beta, label_flipping=True, **default_hyperparam_kwargs
        ),
        "neg": lambda beta: get_clearvae_trainer(
            beta=beta, label_flipping=False, **default_hyperparam_kwargs
        ),
        "bvae": lambda beta: get_clearvae_trainer(
            beta=beta, label_flipping=None, **{**default_hyperparam_kwargs, "alpha": 0}
        ),
        "sepvae": lambda beta: get_sepvae_trainer(
            beta=beta, la=1, **default_hyperparam_kwargs
        ),
        "mlvae": lambda beta: get_hierachical_vae_trainer(
            beta=beta, z_dim=args.z_dim, group_mode="MLVAE", device=args.device
        ),
        "gvae": lambda beta: get_hierachical_vae_trainer(
            beta=beta, z_dim=args.z_dim, group_mode="GVAE", device=args.device
        ),
    }

    results = {model: ExperimentHelper() for model in models}

    # iterate all beta
    for beta in BETAS:
        for model, get_trainer in models.items():
            trainer = get_trainer(beta)
            results[model].run_experiment(
                trainer,
                loaders["train"],
                loaders["valid"],
                loaders["test"],
                trainer_kwargs["epochs"],
            )

    # Create DataFrame
    df_mig_elbo = pd.DataFrame(
        {
            "model": [model for model in models for _ in BETAS],
            "beta": BETAS * len(models),
            "mig": [mig for model in models for mig in results[model].get_results()[0]],
            "elbo": [
                elbo for model in models for elbo in results[model].get_results()[1]
            ],
        }
    )

    df_mig_elbo.to_csv(
        f"./expr_output/cmnist/mig_elbo_s{args.seed}_a{args.alpha}_z{args.z_dim}_t{args.temperature}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
