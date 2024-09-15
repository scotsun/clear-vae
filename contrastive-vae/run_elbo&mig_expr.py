import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
from src.model import VAE
from src.trainer import CDVAETrainer
from src.utils import CMNISTGenerator, CMNIST
from corruption_utils import corruptions

CORRUPTION_FNS = {
    corruptions.identity: 0.2,
    corruptions.stripe: 0.2,
    corruptions.zigzag: 0.3,
    corruptions.canny_edges: 0.1,
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
    generator = CMNISTGenerator(mnist, CORRUPTION_FNS)
    dataset = CMNIST(
        generator, transforms.Compose([transforms.ToTensor(), lambda img: img / 255.0])
    )
    train, valid, test = random_split(dataset, [40_000, 10_000, 10_000])
    return train, valid, test


def get_model_trainer(beta, label_flipping, z_dim, alpha, temperature, device):
    vae = VAE(total_z_dim=z_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)
    trainer = CDVAETrainer(
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
    trainer: CDVAETrainer,
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


def main():
    args = get_args()
    train, valid, test = get_data(args.seed)
    train_loader = DataLoader(train, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=128, shuffle=False)
    test_loader = DataLoader(test, batch_size=128, shuffle=False)

    model_kwargs = {
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

    lf_migs, nlf_migs, bvae_migs = [], [], []
    lf_elbos, nlf_elbos, bvae_elbos = [], [], []
    # iterate all beta
    for beta in BETAS:
        model_kwargs["alpha"] = args.alpha
        lf_trainer = get_model_trainer(beta=beta, label_flipping=True, **model_kwargs)
        nlf_trainer = get_model_trainer(beta=beta, label_flipping=False, **model_kwargs)

        model_kwargs["alpha"] = 0
        bvae_trainer = get_model_trainer(beta=beta, label_flipping=None, **model_kwargs)

        _main_helper(lf_migs, lf_elbos, lf_trainer, **trainer_kwargs)
        _main_helper(nlf_migs, nlf_elbos, nlf_trainer, **trainer_kwargs)
        _main_helper(bvae_migs, bvae_elbos, bvae_trainer, **trainer_kwargs)

    # save as csv
    df_mig_elbo = pd.DataFrame(
        {
            "model": ["lf" for _ in range(len(BETAS))]
            + ["nlf" for _ in range(len(BETAS))]
            + ["bvae" for _ in range(len(BETAS))],
            "beta": BETAS * 3,
            "mig": lf_migs + nlf_migs + bvae_migs,
            "elbo": lf_elbos + nlf_elbos + bvae_elbos,
        }
    )
    df_mig_elbo.to_csv(
        f"""
        ./expr_output/ckmnist/mig_elbo_seed{args.seed}
        _a{args.alpha}_z{args.z_dim}_t{args.temperature}.csv
        """,
        index=False,
    )


if __name__ == "__main__":
    main()
