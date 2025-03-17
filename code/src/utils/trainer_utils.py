"""Trainer utility functions."""

import torch
import torch.nn as nn


from src.models.vae import *  # noqa 403
from src.models.cnn import *  # noqa 403

from src.trainer import (
    SimpleCNNTrainer,
    HierarchicalVAETrainer,
    CLEARVAETrainer,
    ClearTCVAETrainer,
    ClearMIMVAETrainer,
)
from src.models.mi_estimator import *  # noqa 403


def get_cnn_trainer(
    n_class, device, cnn_arch: str = "SimpleCNNClassifier", in_channel: str = 1
):
    cnn = eval(cnn_arch)(n_class=n_class, in_channel=in_channel).to(device)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    trainer = SimpleCNNTrainer(
        cnn, optimizer, criterion, verbose_period=5, device=device
    )
    return trainer


def get_hierarchical_vae_trainer(
    beta, vae_lr, z_dim, group_mode, device, vae_arch: str = "VAE", in_channel: str = 1
):
    vae = eval(vae_arch)(
        total_z_dim=z_dim, in_channel=in_channel, group_mode=group_mode
    ).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=vae_lr)
    trainer = HierarchicalVAETrainer(
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


def get_clearvae_trainer(
    beta,
    label_flipping,
    vae_lr,
    z_dim,
    alpha,
    temperature,
    device,
    vae_arch: str = "VAE",
    in_channel: str = 1,
):
    vae = eval(vae_arch)(total_z_dim=z_dim, in_channel=in_channel).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=vae_lr)
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


def get_cleartcvae_trainer(
    beta,
    la,
    vae_lr,
    factor_cls_lr,
    z_dim,
    alpha,
    temperature,
    device,
    vae_arch: str = "VAE",
    in_channel: str = 1,
):
    vae = eval(vae_arch)(total_z_dim=z_dim, in_channel=in_channel).to(device)
    factor_cls = nn.Sequential(
        nn.Linear(z_dim, z_dim),
        nn.ReLU(),
        nn.Linear(z_dim, 1),
        nn.Sigmoid(),
    ).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=vae_lr)
    factor_optimizer = torch.optim.Adam(factor_cls.parameters(), lr=factor_cls_lr)
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
    beta,
    mi_estimator: str,
    la,
    vae_lr,
    mi_estimator_lr,
    z_dim,
    alpha,
    temperature,
    device,
    vae_arch: str = "VAE",
    in_channel: str = 1,
):
    vae = eval(vae_arch)(total_z_dim=z_dim, in_channel=in_channel).to(device)
    mi_estimator = eval(mi_estimator)(
        x_dim=z_dim // 2, y_dim=z_dim // 2, hidden_size=z_dim
    ).to(device)
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=vae_lr)
    mi_estimator_optimizer = torch.optim.Adam(
        mi_estimator.parameters(), lr=mi_estimator_lr
    )
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
