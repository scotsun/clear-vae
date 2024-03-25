"""Model."""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, z_dim) -> None:
        super().__init__()
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
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_c(h), self.logvar_c(h), self.mu_s(h), self.logvar_s(h)

    def decode(self, z):
        x = self.decoder(z)
        return x

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate(self, mu_c, mu_s, logvar_c, logvar_s, test_verbose=False):
        z_c = self.sample(mu_c, logvar_c)
        z_s = self.sample(mu_s, logvar_s)
        z = torch.cat([z_c, z_s], dim=1)
        if test_verbose:
            print(z[0])
        xhat = self.decode(z)
        return xhat

    def forward(self, x, test_verbose=False):
        mu_c, logvar_c, mu_s, logvar_s = self.encode(x)
        xhat = self.generate(mu_c, logvar_c, mu_s, logvar_s, test_verbose)
        return xhat, mu_c, logvar_c, mu_s, logvar_s
