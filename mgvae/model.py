"""Implementing MGVAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dis


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


def mgvae_loss(x_reconstr, x, mu_minor, logvar_minor, mu_major, logvar_major):
    reconstruction_loss = F.binary_cross_entropy(x_reconstr, x, reduction="mean")
    kl_regularization = kl_lb_loss(mu_minor, logvar_minor, mu_major, logvar_major)
    return reconstruction_loss + kl_regularization


class MGVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.majority_prior_logvar = nn.Parameter(torch.zeros(1))

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


# if __name__ == "__main__":
#     from torchsummary import summary

#     model = MGVAE(784, 64, 8)
#     summary(model, (784,))
