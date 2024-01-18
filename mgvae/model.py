"""Implementing MGVAE & EWC."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dis
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset, Subset, DataLoader


def log_mg_prior(z, mu_major, logvar_major):
    """Calculate the majority-guided prior `log r(z|x+)`."""
    # note: z.shape = (T, B, m), mu_major.shape = (N, m)
    N, m = mu_major.shape
    r_dist = dis.MultivariateNormal(mu_major, logvar_major.exp() * torch.eye(m))

    z = z[:, :, None, :].repeat(1, 1, N, 1)
    r_logpdf = r_dist.log_prob(z)

    dims = len(r_logpdf.shape)
    log_prior_pdf = r_logpdf.exp().mean(dim=dims - 1).log()
    # output.shape = (T, B)
    return log_prior_pdf


def kl_estimated_loss(
    mu_minor, logvar_minor, mu_major, logvar_major, device: str, reduce: bool, T=30
):
    """Calculate KL loss using MC estimation."""
    B, _ = mu_minor.shape

    q = dis.MultivariateNormal(mu_minor, logvar_minor.exp().diag_embed())
    z = q.sample(sample_shape=(T,)).to(device)
    # z.shape is (T, B, m)
    # define: ratio = prior_pdf / encoder_pdf, see details at http://joschu.net/blog/kl-approx.html
    # estimate kl per (mu(x-), logvar(x-)) by the sample mean of (ratio - 1) - log_ratio
    log_ratio = q.log_prob(z) - log_mg_prior(z, mu_major, logvar_major)
    # log_ratio.shape = (T, B)
    kl_est = log_ratio.mean(dim=0)

    return kl_est.mean() if reduce else kl_est


def kl_ub_loss(mu_minor, logvar_minor, mu_major, logvar_major, reduce: bool):
    """Calculate KL loss's upper bound using another Jensen's Inequality / GM-AM Inequality."""
    B, m = mu_minor.shape
    N, _ = mu_major.shape

    mu_minor = mu_minor.T[None, :, :].repeat(N, 1, 1)
    mu_major = mu_major[:, :, None].repeat(1, 1, B)
    kl_variant = (
        0.5 * ((mu_minor - mu_major) ** 2).sum(dim=1).mean(dim=0) / logvar_major.exp()
    )

    kl_invariant = 0.5 * (
        m * logvar_major
        - logvar_minor.sum(dim=1)
        - m
        + logvar_minor.exp().sum(dim=1) / logvar_major.exp()
    )

    kl = kl_invariant + kl_variant

    return kl.mean() if reduce else kl


def mgvae_loss(
    x_reconstr,
    x,
    mu_minor,
    logvar_minor,
    mu_major,
    logvar_major,
    kl_loss_fn: str,
    device: str,
    reduce: bool = True,
):
    """
    Calculate the two parts of the loss objective.

    `reduce` (reduction to mean) is set to `True` by default.
    One would set it to `False` for calculating fisher information in EWC.
    """
    n_dims = len(x.shape)
    _reconstruction_loss = F.binary_cross_entropy(x_reconstr, x, reduction="none").sum(
        dim=list(range(n_dims))[1:]
    )
    reconstruction_loss = (
        _reconstruction_loss.mean() if reduce else _reconstruction_loss
    )

    match kl_loss_fn:
        case "estimated":
            kl_regularization = kl_estimated_loss(
                mu_minor, logvar_minor, mu_major, logvar_major, device, reduce
            )
        case "upper_bound":
            kl_regularization = kl_ub_loss(
                mu_minor, logvar_minor, mu_major, logvar_major, reduce
            )
        case _:
            raise ValueError("incorrect 'kl_loss_fn' value.")
    return reconstruction_loss, kl_regularization


class MGVAE(nn.Module):
    """Majority-Guided VAE."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        majority_data: Dataset,
        N_maj: int,
        device,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.majority_data = majority_data
        self.N_maj = N_maj
        self.device = device

        self.logvar_major = nn.Parameter(torch.ones(1))

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
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
        return x_hat, mu, logvar

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
                X_batch = X_batch.view(-1, self.input_dim).to(self.device)
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
        """Fine-tune with EWC."""
        print("finetuning starts:")
        # Fisher information calculated based on the pre-trained model
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
                    X_batch = X_batch.view(-1, self.input_dim).to(self.device)
                    optimizer.zero_grad()
                    Xh_batch, mu_batch, logvar_batch = self(X_batch)
                    # get a sub-sample of majorities to compute prior
                    mu_major = self.compute_mg_prior_mu()
                    recon_loss, kl_loss = mgvae_loss(
                        Xh_batch,
                        X_batch,
                        mu_batch,
                        logvar_batch,
                        mu_major,
                        self.logvar_major,
                        kl_loss_fn,
                        self.device,
                    )
                    ewc_loss = ewc.penalty(self)
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

    def pretrain(
        self,
        train_loader: DataLoader,
        kl_loss_fn: str,
        lr,
        epochs,
        verbose_period,
    ):
        """Pre-train."""
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
                    X_batch = X_batch.view(-1, self.input_dim).to(self.device)
                    optimizer.zero_grad()
                    Xh_batch, mu_batch, logvar_batch = self(X_batch)
                    # get a sub-sample of majorities to compute prior
                    mu_major = self.compute_mg_prior_mu()
                    recon_loss, kl_loss = mgvae_loss(
                        Xh_batch,
                        X_batch,
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
        x = (
            sample_dataset.dataset.tensors[0][sample_dataset.indices]
            .view(-1, self.model.input_dim)
            .to(self.device)
        )
        # foward pass using mgvae_loss
        xh, mu, logvar = self.model(x)
        recon_loss, kl_loss = mgvae_loss(
            xh,
            x,
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
