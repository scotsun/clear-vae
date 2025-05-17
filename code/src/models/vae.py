"""VAE Models."""

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(
        self, total_z_dim, in_channel: int = 1, group_mode: str | None = None
    ) -> None:
        super().__init__()
        self.mode = group_mode
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

    def sample(self, mu, logvar):
        """Reparameterization."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def generate(
        self, mu_c, logvar_c, mu_s, logvar_s, g_dict: dict | None = None, explicit=False
    ):
        """
        g_dict: if provided ML-VAE or GVAE will have evi. acc.
        explicit: bool indicating if return generated latent z
        """
        if g_dict is None:
            z_c = self.sample(mu_c, logvar_c)
        else:
            z_c, _, _ = groupwise_reparam_each(mu_c, logvar_c, g_dict)
        z_s = self.sample(mu_s, logvar_s)
        z = torch.cat([z_c, z_s], dim=-1)
        xhat = self.decode(z)
        if explicit:
            return xhat, z
        else:
            return xhat

    def forward(self, x, label=None, explicit=False) -> tuple:
        mu_c, logvar_c, mu_s, logvar_s = self.encode(x)

        if label is not None:  # if label is provided, then we have a grouping dict
            mu_c, logvar_c, g_dict = accumulate_group_evidence(
                mu_c, logvar_c, label, mode=self.mode
            )
        else:  # else we do not have a grouping dict
            g_dict = None

        latent_params = {
            "mu_c": mu_c,
            "logvar_c": logvar_c,
            "mu_s": mu_s,
            "logvar_s": logvar_s,
        }
        if explicit:
            xhat, z = self.generate(mu_c, logvar_c, mu_s, logvar_s, g_dict, True)
            return xhat, latent_params, z  # z = (z_c, z_s)
        else:
            xhat = self.generate(mu_c, logvar_c, mu_s, logvar_s, g_dict, False)
            return xhat, latent_params


class VAE64(VAE):
    def __init__(
        self, total_z_dim, in_channel: int = 3, group_mode: str | None = None
    ) -> None:
        super().__init__(total_z_dim, in_channel, group_mode)

        self.z_dim = int(total_z_dim / 2)
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
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
            nn.Unflatten(1, (512, 2, 2)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, 0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, 0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channel, 4, 2, 1, 0),
            nn.BatchNorm2d(in_channel),
            nn.Sigmoid(),
        )


def accumulate_group_evidence(
    mu_c, logvar_c, label_batch, mode: str
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    device = mu_c.device
    groups = (label_batch).unique(sorted=True)
    mu_acc_grp = torch.zeros(len(groups), mu_c.size(1)).to(device)
    logvar_acc_grp = torch.zeros(len(groups), logvar_c.size(1)).to(device)

    group_idx = {}
    for i, g in enumerate(groups):
        group_label = g.item()
        group_select = label_batch.eq(group_label)

        group_idx[g.item()] = group_select.nonzero().view(-1)

        if mode == "MLVAE":
            loginvvar_c = -logvar_c[group_select, :]
            group_mu_invvar = (mu_c[group_select, :] * loginvvar_c.exp()).sum(dim=0)
            group_loginvvar = loginvvar_c.logsumexp(dim=0)

            mu_acc_grp[i] = group_mu_invvar * torch.exp(-group_loginvvar)
            logvar_acc_grp[i] = -group_loginvvar
        elif mode == "GVAE":
            mu_acc_grp[i] = mu_c[group_select, :].mean(dim=0)
            # group_size <- group_select.sum()
            logvar_acc_grp[i] = (
                logvar_c[group_select, :].logsumexp(dim=0) - group_select.sum().log()
            )
        else:
            raise NotImplementedError("only support using MLVAE or GVAE")

    return mu_acc_grp, logvar_acc_grp, group_idx


def groupwise_reparam_each(mu_acc_grp, logvar_acc_grp, g_idx: dict):
    device = mu_acc_grp.device

    # batch_size = sum(g_idx.values())
    std_acc_grp = torch.exp(0.5 * logvar_acc_grp)

    z_grps = []
    indices = []
    sizes = []
    for i, (g, idx) in enumerate(g_idx.items()):
        n = len(idx)
        if n == 0:  # skip for empty class
            pass
        # eps = torch.randn(1, std_acc_grp.size(1)).repeat(n, 1).to(device)
        eps = torch.randn(n, std_acc_grp.size(1)).to(device)
        z_grp = mu_acc_grp[i][None, :] + eps * std_acc_grp[i][None, :]

        z_grps.append(z_grp)
        indices.append(idx)
        sizes.append(torch.ones_like(idx) * len(idx))

    z_grps = torch.cat(z_grps, dim=0)
    indices = torch.cat(indices, dim=0)
    sizes = torch.cat(sizes, dim=0)

    # undo the ordering
    inverse_indices = torch.zeros_like(indices)
    inverse_indices[indices] = torch.arange(len(indices)).to(indices.device)
    z_grps = z_grps[inverse_indices]

    return z_grps, indices, sizes
