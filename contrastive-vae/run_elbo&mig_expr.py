import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from src.model import VAE
from src.trainer import CLEARVAETrainer, MLVAETrainer
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
BETAS = [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=101, help="random seed; default 101"
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


def get_mlvae_trainer(beta, z_dim, device):
    vae = VAE(total_z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)
    trainer = MLVAETrainer(
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


def _main_helper(
    mig_list: list,
    elbo_list: list,
    trainer: CLEARVAETrainer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
):
    trainer.fit(epochs, train_loader, valid_loader)
    print("")
    mig, elbo = trainer.evaluate(test_loader, False, 0)
    mig_list.append(mig)
    elbo_list.append(elbo)
    return


def _main_helper_mlvae(
    mig_list: list,
    elbo_list: list,
    trainer: MLVAETrainer,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    with_evidence_acc: bool,
):
    trainer.fit(epochs, train_loader, valid_loader, with_evidence_acc=True)
    print("")
    mig, elbo = trainer.evaluate(test_loader, False, 0, with_evidence_acc)
    mig_list.append(mig)
    elbo_list.append(elbo)
    return


def main():
    args = get_args()
    train, valid, test = get_data(args.seed)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=128, shuffle=False)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)

    hyperparam_kwargs = {
        "z_dim": args.z_dim,
        "alpha": args.alpha,
        "temperature": args.temperature,
        "device": args.device,
    }
    trainer_kwargs = {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "epochs": 31,
    }

    ps_migs, nps_migs, bvae_migs, ml_acc_migs, ml_nacc_migs = [], [], [], [], []
    ps_elbos, nps_elbos, bvae_elbos, ml_acc_elbos, ml_nacc_elbos = [], [], [], [], []
    # iterate all beta
    for beta in BETAS:
        hyperparam_kwargs["alpha"] = args.alpha
        ps_trainer = get_clearvae_trainer(
            beta=beta, label_flipping=True, **hyperparam_kwargs
        )
        nps_trainer = get_clearvae_trainer(
            beta=beta, label_flipping=False, **hyperparam_kwargs
        )
        bvae_hyperparam_kwargs = hyperparam_kwargs.copy()
        bvae_hyperparam_kwargs["alpha"] = 0
        bvae_trainer = get_clearvae_trainer(
            beta=beta, label_flipping=None, **bvae_hyperparam_kwargs
        )
        ml_acc_trainer = get_mlvae_trainer(
            beta=beta, z_dim=args.z_dim, device=args.device
        )
        ml_nacc_trainer = get_mlvae_trainer(
            beta=beta, z_dim=args.z_dim, device=args.device
        )

        _main_helper(ps_migs, ps_elbos, ps_trainer, **trainer_kwargs)
        _main_helper(nps_migs, nps_elbos, nps_trainer, **trainer_kwargs)
        _main_helper(bvae_migs, bvae_elbos, bvae_trainer, **trainer_kwargs)
        ml_acc_trainer_kwargs = trainer_kwargs.copy()
        ml_acc_trainer_kwargs["with_evidence_acc"] = True
        _main_helper_mlvae(
            ml_acc_migs, ml_acc_elbos, ml_acc_trainer, **ml_acc_trainer_kwargs
        )
        ml_nacc_trainer_kwargs = trainer_kwargs.copy()
        ml_nacc_trainer_kwargs["with_evidence_acc"] = False
        _main_helper_mlvae(
            ml_nacc_migs, ml_nacc_elbos, ml_nacc_trainer, **ml_nacc_trainer_kwargs
        )

    # save as csv
    df_mig_elbo = pd.DataFrame(
        {
            "model": ["ps" for _ in range(len(BETAS))]
            + ["nps" for _ in range(len(BETAS))]
            + ["bvae" for _ in range(len(BETAS))]
            + ["ml_acc" for _ in range(len(BETAS))]
            + ["ml_nacc" for _ in range(len(BETAS))],
            "beta": BETAS * 5,
            "mig": ps_migs + nps_migs + bvae_migs + ml_acc_migs + ml_nacc_migs,
            "elbo": ps_elbos + nps_elbos + bvae_elbos + ml_acc_elbos + ml_nacc_elbos,
        }
    )
    df_mig_elbo.to_csv(
        f"./expr_output/cmnist/mig_elbo_s{args.seed}_a{args.alpha}_z{args.z_dim}_t{args.temperature}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
