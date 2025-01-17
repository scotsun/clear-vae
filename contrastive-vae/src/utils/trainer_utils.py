"""Trainer utility functions."""

import torch
import torch.nn as nn


from src.models.vae import VAE
from src.models.cnn import SimpleCNNClassifier

from src.trainer import (
    SimpleCNNTrainer,
    HierachicalVAETrainer,
    CLEARVAETrainer,
    ClearTCVAETrainer,
    ClearMIMVAETrainer,
)
from src.models.mi_estimator import *  # noqa 403


def get_cnn_trainer(n_class, device):
    cnn = SimpleCNNClassifier(n_class=n_class).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = SimpleCNNTrainer(
        cnn, optimizer, criterion, verbose_period=5, device=device
    )
    return trainer


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
            "alpha": alpha,
            "beta": beta,
            "label_flipping": label_flipping,
            "loc": 0,
            "scale": 1,
        },
        verbose_period=5,
        device=device,
    )
    return trainer


def get_cleartcvae_trainer(beta, la, z_dim, alpha, temperature, device):
    vae = VAE(total_z_dim=z_dim).to(device)
    factor_cls = nn.Sequential(
        nn.Linear(z_dim, z_dim),
        nn.ReLU(),
        nn.Linear(z_dim, 1),
        nn.Sigmoid(),
    ).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)
    factor_optimizer = torch.optim.Adam(factor_cls.parameters(), lr=1e-4)
    trainer = ClearTCVAETrainer(
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


def get_clearmimvae_trainer(
    beta, mi_estimator: str, la, z_dim, alpha, temperature, device
):
    vae = VAE(total_z_dim=z_dim).to(device)
    mi_estimator = eval(mi_estimator)(x_dim=8, y_dim=8, hidden_size=16).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=5e-4)
    mi_estimator_optimizer = torch.optim.Adam(mi_estimator.parameters(), lr=1e-4)
    trainer = ClearMIMVAETrainer(
        vae,
        mi_estimator,
        optimizers={
            "vae_optim": vae_optimizer,
            "mi_estimator_optim": mi_estimator_optimizer,
        },
        sim_fn="cosine",
        hyperparameter={
            "temperature": temperature,
            "beta": beta,
            "loc": 0,
            "scale": 1,
            "alpha": alpha,
            "lambda": la,
        },
        verbose_period=5,
        device=device,
    )
    return trainer
