"""Model."""

import torch
import torch.nn as nn


class SimpleCNNClassifier(nn.Module):
    def __init__(self, n_class: int = 10, in_channel: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, n_class),
        )

    def forward(self, x):
        return self.net(x)


class VAE(nn.Module):
    def __init__(self, total_z_dim, in_channel: int = 1) -> None:
        super().__init__()
        z_dim = int(total_z_dim / 2)
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, 3, 2, 1),
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
        self.mu_c = nn.Linear(2048, z_dim)
        self.logvar_c = nn.Linear(2048, z_dim)
        self.mu_s = nn.Linear(2048, z_dim)
        self.logvar_s = nn.Linear(2048, z_dim)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_dim * 2, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channel, 3, 2, 1, 1),
            nn.BatchNorm2d(in_channel),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_c(h), self.logvar_c(h), self.mu_s(h), self.logvar_s(h)

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample(self, mu, log_var):
        """Reparameterization."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate(self, mu_c, logvar_c, mu_s, logvar_s, explicit=False):
        """
        explicit: bool indicating if return generated latent z
        """
        z_c = self.sample(mu_c, logvar_c)
        z_s = self.sample(mu_s, logvar_s)
        z = torch.cat([z_c, z_s], dim=-1)
        xhat = self.decode(z)
        if explicit:
            return xhat, z
        else:
            return xhat

    def forward(self, x, explicit=False) -> tuple:
        mu_c, logvar_c, mu_s, logvar_s = self.encode(x)
        latent_param = {
            "mu_c": mu_c,
            "logvar_c": logvar_c,
            "mu_s": mu_s,
            "logvar_s": logvar_s,
        }
        if explicit:
            xhat, z = self.generate(mu_c, logvar_c, mu_s, logvar_s, True)
            return xhat, latent_param, z
        else:
            xhat = self.generate(mu_c, logvar_c, mu_s, logvar_s, False)
            return xhat, latent_param


def interpolate_latent(latent1, latent2, num_steps, device):
    """Interpolate between two latent vectors."""
    p = torch.linspace(1, 0, num_steps).to(device)
    # reshape to interpolation matrix shape (num_step, latent_dim)
    latent_dim = latent1.shape[-1]
    p = p[:, None].repeat((1, latent_dim))
    latent1 = latent1[None, :].repeat((num_steps, 1))
    latent2 = latent2[None, :].repeat((num_steps, 1))
    # generate interpolation matrix
    return p * latent1 + (1 - p) * latent2
