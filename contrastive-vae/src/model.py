"""Model."""

import torch
import torch.nn as nn
from torch.autograd import Variable


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
        )
        self.clf_head = nn.Linear(2048, n_class)

    def forward(self, x):
        h = self.net(x)
        return self.clf_head(h)


class VAE(nn.Module):
    def __init__(self, total_z_dim, in_channel: int = 1) -> None:
        super().__init__()
        self.z_dim = int(total_z_dim / 2)
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
        self.mu_c = nn.Linear(2048, self.z_dim)
        self.logvar_c = nn.Linear(2048, self.z_dim)
        self.mu_s = nn.Linear(2048, self.z_dim)
        self.logvar_s = nn.Linear(2048, self.z_dim)
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim * 2, 2048),
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

    def sample_by_group(self, group_mu, group_logvar, label):
        "Reparameterization by group"
        device = group_mu.device
        eps_dict = {}
        for _l in torch.unique(label):
            eps_dict[_l.item()] = torch.randn(1, self.z_dim, device=device)

        group_eps = torch.FloatTensor(group_mu.size(0), group_mu.size(1)).to(device)
        for i in range(group_mu.size(0)):
            group_eps[i] = eps_dict[label[i].item()]

        group_std = torch.exp(0.5 * group_logvar)
        return group_mu + group_eps * group_std

    def generate(self, mu_c, logvar_c, mu_s, logvar_s, label=None, explicit=False):
        """
        label: if provided the VAE -> ML-VAE
        explicit: bool indicating if return generated latent z
        """
        if label is None:
            z_c = self.sample(mu_c, logvar_c)
        else:
            # TODO:
            z_c = self.sample_by_group(mu_c, logvar_c, label)
            # z_c = self.sample(mu_c, logvar_c)
        z_s = self.sample(mu_s, logvar_s)
        z = torch.cat([z_c, z_s], dim=-1)
        xhat = self.decode(z)
        if explicit:
            return xhat, z
        else:
            return xhat

    def forward(self, x, label=None, explicit=False) -> tuple:
        mu_c, logvar_c, mu_s, logvar_s = self.encode(x)

        if label is not None:  # if label is provided
            mu_c, logvar_c = accumulate_group_evidence(
                mu_c, logvar_c, label, mu_c.device
            )

        latent_params = {
            "mu_c": mu_c,
            "logvar_c": logvar_c,
            "mu_s": mu_s,
            "logvar_s": logvar_s,
        }
        if explicit:
            xhat, z = self.generate(mu_c, logvar_c, mu_s, logvar_s, label, True)
            return xhat, latent_params, z  # z = (z_c, z_s)
        else:
            xhat = self.generate(mu_c, logvar_c, mu_s, logvar_s, label, False)
            return xhat, latent_params


def accumulate_group_evidence(mu_c, logvar_c, labels_batch, device):
    var_dict = {}
    mu_dict = {}

    # convert logvar to variance for calculations
    var_c = logvar_c.exp_()

    # calculate var inverse for each group using group vars
    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()

        # remove 0 values from variances
        var_c[i][var_c[i] == float(0)] = 1e-8

        if group_label in var_dict.keys():
            var_dict[group_label] += 1 / var_c[i]
        else:
            var_dict[group_label] = 1 / var_c[i]

    # invert var inverses to calculate mu and return value
    for group_label in var_dict.keys():
        var_dict[group_label] = 1 / var_dict[group_label]

    # calculate mu for each group
    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()

        if group_label in mu_dict.keys():
            mu_dict[group_label] += mu_c[i] * (1 / var_c[i])
        else:
            mu_dict[group_label] = mu_c[i] * (1 / var_c[i])

    # multiply group var with sums calculated above to get mu for the group
    for group_label in mu_dict.keys():
        mu_dict[group_label] *= var_dict[group_label]

    # replace individual mu and logvar values for each sample with group mu and logvar
    group_mu = torch.FloatTensor(mu_c.size(0), mu_c.size(1)).to(device)
    group_var = torch.FloatTensor(var_c.size(0), var_c.size(1)).to(device)

    for i in range(len(labels_batch)):
        group_label = labels_batch[i].item()
        group_mu[i] = mu_dict[group_label]
        group_var[i] = var_dict[group_label]
        # remove 0 from var before taking log
        group_var[i][group_var[i] == float(0)] = 1e-8

    # convert group vars into logvars before returning
    return Variable(group_mu, requires_grad=True), Variable(
        torch.log(group_var), requires_grad=True
    )
