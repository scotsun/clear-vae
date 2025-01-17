import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from corruption_utils import corruptions

from src.utils.trainer_utils import (
    get_clearvae_trainer,
    get_cleartcvae_trainer,
    get_clearmimvae_trainer,
    get_hierachical_vae_trainer,
)
from src.trainer import HierachicalVAETrainer
from src.utils.data_utils import StyledMNISTGenerator, StyledMNIST


CORRUPTION_FNS = {
    corruptions.identity: 0.15,
    corruptions.stripe: 0.2,
    corruptions.zigzag: 0.25,
    corruptions.canny_edges: 0.1,
    lambda x: corruptions.scale(x, 5): 0.1,
    corruptions.brightness: 0.2,
}
# BETAS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]
BETAS = [1 / 8]


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
        "clear-ps": lambda beta: get_clearvae_trainer(
            beta=beta, label_flipping=True, **default_hyperparam_kwargs
        ),
        "clear-neg": lambda beta: get_clearvae_trainer(
            beta=beta, label_flipping=False, **default_hyperparam_kwargs
        ),
        "bvae": lambda beta: get_clearvae_trainer(
            beta=beta, label_flipping=None, **{**default_hyperparam_kwargs, "alpha": 0}
        ),
        "clear-tc": lambda beta: get_cleartcvae_trainer(
            beta=beta, la=1, **default_hyperparam_kwargs
        ),
        "clear-mim (L1OutUB)": lambda beta: get_clearmimvae_trainer(
            beta=beta, mi_estimator="L1OutUB", la=3, **default_hyperparam_kwargs
        ),
        "clear-mim (CLUB-S)": lambda beta: get_clearmimvae_trainer(
            beta=beta, mi_estimator="CLUBSample", la=3, **default_hyperparam_kwargs
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
        print(
            f"""
            ************************************************\n
            BETA: {beta} \n
            ************************************************\n
        """
        )
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
        f"./expr_output/styled-mnist/mig_elbo_s{args.seed}_a{args.alpha}_z{args.z_dim}_t{args.temperature}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
