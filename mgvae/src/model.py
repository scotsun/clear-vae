"""Implementing MGVAE & EWC."""

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset, Subset, DataLoader
from .losses import mgvae_loss


class MGVAE(nn.Module):
    """Majority-Guided VAE."""

    def __init__(
        self,
        latent_dim,
        majority_data: Dataset,
        N_maj: int,
        device,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.majority_data = majority_data
        self.N_maj = N_maj
        self.device = device

        self.logvar_major = nn.Parameter(torch.tensor(1.0))
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std)
        z = mu + std * noise
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar, z

    def compute_mg_prior_mu(self):
        """Calculate a majority guided prior using a sub-sample of the majority set."""
        majority_data = self.majority_data
        majority_dl = DataLoader(
            Subset(
                majority_data,
                np.random.choice(len(majority_data), self.N_maj, replace=False),
            ),
            batch_size=128,
            shuffle=True,
        )
        mu_major = torch.zeros(self.N_maj, self.latent_dim).to(self.device)
        idx_cursor = 0
        with torch.no_grad():
            for X_batch, _ in majority_dl:
                X_batch = X_batch.to(self.device)
                mu_batch, _ = self.encode(X_batch)
                mu_major[idx_cursor : (idx_cursor + len(X_batch)), :] = mu_batch
                idx_cursor += len(X_batch)
        return mu_major

    def finetune(
        self,
        train_loader: DataLoader,
        kl_loss_fn: str,
        ewc_lambda,
        lr,
        epochs,
        verbose_period,
    ):
        """
        Fine-tuning with EWC.

        `kl_loss_fn` is in {`upper_bound`, `estimated`}.
        """
        print("finetuning starts:")
        # Fisher information calculated based on the pre-trained model
        if ewc_lambda > 0:
            ewc = EWC(self, self.majority_data, kl_loss_fn, self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            total_recon_loss, total_kl_loss, total_ewc_loss = 0.0, 0.0, 0.0
            total_loss = 0.0
            total_batch = 0.0
            verbose = (epoch % verbose_period) == 0
            with tqdm(
                train_loader, unit="batch", mininterval=0, disable=not verbose
            ) as bar:
                bar.set_description(f"Epoch {epoch}")
                for X_batch, _ in bar:
                    X_batch = X_batch.to(self.device)
                    optimizer.zero_grad()
                    Xh_batch, mu_batch, logvar_batch, z_batch = self(X_batch)
                    # get a sub-sample of majorities to compute prior
                    mu_major = self.compute_mg_prior_mu()
                    recon_loss, kl_loss = mgvae_loss(
                        Xh_batch,
                        X_batch,
                        z_batch,
                        mu_batch,
                        logvar_batch,
                        mu_major,
                        self.logvar_major,
                        kl_loss_fn,
                        self.device,
                    )
                    if ewc_lambda > 0:
                        ewc_loss = ewc.penalty(self)
                    else:
                        ewc_loss = torch.tensor(0.0).to(self.device)
                    loss = recon_loss + kl_loss + ewc_lambda * ewc_loss

                    loss.backward()
                    optimizer.step()
                    # update progress
                    total_recon_loss += recon_loss.item()
                    total_kl_loss += kl_loss.item()
                    total_ewc_loss += ewc_loss.item()
                    total_loss += loss.item()
                    total_batch += 1

                    bar.set_postfix(
                        recon_loss=float(total_recon_loss / total_batch),
                        kl_loss=float(total_kl_loss / total_batch),
                        ewc_loss=float(total_ewc_loss / total_batch),
                        total=float(total_loss / total_batch),
                    )
            if verbose:
                print(self.logvar_major.data)

    def pretrain(
        self,
        train_loader: DataLoader,
        kl_loss_fn: str,
        lr,
        epochs,
        verbose_period,
    ):
        """
        Pre-training.

        `kl_loss_fn` is in {`upper_bound`, `estimated`}.
        """
        print("pre-training starts:")
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            total_recon_loss, total_kl_loss = 0.0, 0.0
            total_loss = 0.0
            total_batch = 0.0
            verbose = (epoch % verbose_period) == 0
            with tqdm(
                train_loader, unit="batch", mininterval=0, disable=not verbose
            ) as bar:
                bar.set_description(f"Epoch {epoch}")
                for X_batch, _ in bar:
                    X_batch = X_batch.to(self.device)
                    optimizer.zero_grad()
                    Xh_batch, mu_batch, logvar_batch, z_batch = self(X_batch)
                    # get a sub-sample of majorities to compute prior
                    mu_major = self.compute_mg_prior_mu()
                    recon_loss, kl_loss = mgvae_loss(
                        Xh_batch,
                        X_batch,
                        z_batch,
                        mu_batch,
                        logvar_batch,
                        mu_major,
                        self.logvar_major,
                        kl_loss_fn,
                        self.device,
                    )
                    loss = recon_loss + kl_loss
                    loss.backward()
                    optimizer.step()
                    # update progress
                    total_recon_loss += recon_loss.item()
                    total_kl_loss += kl_loss.item()
                    total_loss += loss.item()
                    total_batch += 1

                    bar.set_postfix(
                        recon_loss=float(total_recon_loss / total_batch),
                        kl_loss=float(total_kl_loss / total_batch),
                        total=float(total_loss / total_batch),
                    )
            if verbose:
                print(self.logvar_major.data)


class EWC:
    """Elastic Weight Consolidation (specific to the MGVAE model)."""

    def __init__(
        self,
        model: MGVAE,
        dataset: TensorDataset,
        kl_loss_fn: str,
        device: str,
        N: int = 200,
    ) -> None:
        """
        Initialize an elastic weight consolidation object.

        The dataset (old task) used in the init will be downsampled and used to calculate the Monte Carlo estimates for Fisher Information.
        The param importance calculated based on the data from the "old task"
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.params_old = dict()
        for n, p in self.model.named_parameters():
            self.params_old[n] = p.clone()
        self.fisher_information: dict = self._compute_fisher(kl_loss_fn, N)

    def _compute_fisher(self, kl_loss_fn: str, N):
        """Calculate diagonal elements in the information matrix."""
        # retrieve a subset data
        sample_dataset = Subset(
            self.dataset, np.random.choice(len(self.dataset), N, replace=False)
        )
        x = next(iter(DataLoader(sample_dataset, N, shuffle=True)))[0].to(self.device)

        # foward pass using mgvae_loss
        xh, mu, logvar, z = self.model(x)
        recon_loss, kl_loss = mgvae_loss(
            xh,
            x,
            z,
            mu,
            logvar,
            self.model.compute_mg_prior_mu(),
            self.model.logvar_major,
            kl_loss_fn=kl_loss_fn,
            device=self.device,
            reduce=False,
        )
        # calculate fisher information using gradients
        fisher_information = dict()
        for n, p in self.model.named_parameters():
            fisher_information[n] = torch.zeros_like(p)
        for i in range(N):
            gradients = torch.autograd.grad(
                recon_loss[i] + kl_loss[i], self.model.parameters(), retain_graph=True
            )
            for j, n in enumerate(self.params_old):
                fisher_information[n] += gradients[j] ** 2 / N

        return fisher_information

    def penalty(self, updated_model: nn.Module):
        """Sum over all F(param_new - param_old)^2."""
        loss = 0
        for n, param_new in updated_model.named_parameters():
            loss += (
                self.fisher_information[n] * (param_new - self.params_old[n]) ** 2
            ).sum()
        return loss
