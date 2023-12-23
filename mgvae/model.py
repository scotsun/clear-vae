"""Implementing MGVAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dis
from tqdm import tqdm

from torch.utils.data import DataLoader


def log_mg_prior(z, mu_major, logvar_major):
    # z.shape = (T, m)
    T, _ = z.shape
    N, m = mu_major.shape

    mu_major = mu_major[None, :, :].repeat(T, 1, 1)
    z = z[:, None, :].repeat(1, N, 1)
    r_dist = dis.MultivariateNormal(mu_major, logvar_major.exp() * torch.eye(m))
    r_logpdf = r_dist.log_prob(z)
    log_prior_pdf = r_logpdf.exp().mean(dim=1).log()
    return log_prior_pdf


def kl_estimated_loss(mu_minor, logvar_minor, mu_major, logvar_major, T=50):
    B, _ = mu_minor.shape
    kl_loss = 0.0
    for i in range(B):
        q = dis.MultivariateNormal(mu_minor[i], logvar_minor[i].exp().diag())
        z = q.sample(sample_shape=(T,))
        # ratio = prior_pdf / encoder_pdf
        # estimate kl per (mu(x-), logvar(x-)) via sample mean of (ratio - 1) - log_ratio
        log_ratio = log_mg_prior(z, mu_major, logvar_major.exp()) - q.log_prob(z)
        kl_est = ((log_ratio.exp() - 1) - log_ratio).mean()
        kl_loss += kl_est / B
    return kl_loss


def kl_lb_loss(mu_minor, logvar_minor, mu_major, logvar_major):
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
    return kl.mean()


def mgvae_loss(
    x_reconstr, x, mu_minor, logvar_minor, mu_major, logvar_major, kl_loss_fn: str
):
    n_dims = len(x.shape)
    reconstruction_loss = (
        F.binary_cross_entropy(x_reconstr, x, reduction="none")
        .sum(dim=list(range(n_dims))[1:])
        .mean()
    )
    match kl_loss_fn:
        case "estimated":
            kl_regularization = kl_estimated_loss(
                mu_minor, logvar_minor, mu_major, logvar_major
            )
        case "lower_bound":
            kl_regularization = kl_lb_loss(
                mu_minor, logvar_minor, mu_major, logvar_major
            )
        case _:
            raise ValueError("incorrect 'kl_loss_fn' value.")
    return reconstruction_loss, kl_regularization


class MGVAE(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, latent_dim, majority_dataloader: DataLoader
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.majority_dataloader = majority_dataloader

        self.logvar_major = nn.Parameter(torch.zeros(1))

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
        N = len(self.majority_dataloader.dataset)
        mu_major = torch.zeros(N, self.latent_dim)
        idx_cursor = 0
        with torch.no_grad():
            for X_batch, _ in self.majority_dataloader:
                mu_batch = self.encode(X_batch)
                mu_major[idx_cursor : (idx_cursor + len(X_batch)), :] = mu_batch
                idx_cursor += len(X_batch)
        return mu_major

    def evaluate(self, dataloader):
        # TODO:
        pass

    def train(self, train_loader, kl_loss_fn, lr, epochs, verbose_period):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(epochs):
            total_recon_loss, total_kl_loss = 0.0, 0.0
            total_batch = 0.0
            verbose = (epoch % verbose_period) == 0
            with tqdm(
                train_loader, unit="batch", miniterval=0, disable=not verbose
            ) as bar:
                bar.set_description(f"Epoch {epoch}")
                for X_batch, _ in bar:
                    optimizer.zero_grad()
                    mu_major = self.compute_mg_prior_mu()
                    Xh_batch, mu_batch, logvar_batch = self(X_batch)
                    recon_loss, kl_loss = mgvae_loss(
                        Xh_batch,
                        X_batch,
                        mu_batch,
                        logvar_batch,
                        mu_major,
                        self.logvar_major,
                        kl_loss_fn,
                    )
                    (recon_loss + kl_loss).backward()
                    optimizer.step()
                    # update progress
                    total_recon_loss += recon_loss.item()
                    total_kl_loss += kl_loss.item()
                    total_batch += 1
                    bar.set_postfix(
                        recon_loss=float(total_recon_loss / total_batch),
                        kl_loss=float(total_kl_loss / total_batch),
                    )


# if __name__ == "__main__":
#     from torchsummary import summary

#     model = MGVAE(784, 64, 8)
#     summary(model, (784,))
