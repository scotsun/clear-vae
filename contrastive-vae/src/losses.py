"""Different loss & divergence functions."""

import torch
import torch.nn.functional as F


def vae_loss(x_reconstr, x, mu_c, mu_s, logvar_c, logvar_s):
    """
    VAE loss with separating factors.
    """
    dims = len(x.shape)
    reconstruction_loss = (
        F.mse_loss(
            x_reconstr, x, reduction="none"
        )  # shoud be mse but bce gives better result
        .sum(dim=list(range(dims))[1:])
        .mean()
    )
    kl_c = -0.5 * torch.mean(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
    kl_s = -0.5 * torch.mean(1 + logvar_s - mu_s.pow(2) - logvar_s.exp())
    return reconstruction_loss + kl_c + kl_s


def divergence_fn(mu_b_c, mu_p_c, logvar_b_c, logvar_p_c, metric="mahalanobis"):
    """
    Divergence between latent distns of image pair.
    """
    if metric == "mahalanobis":
        d = (mu_b_c - mu_p_c) ** 2 / torch.exp(0.5 * (logvar_b_c + logvar_p_c))
        return d.mean(dim=1)
    else:
        raise NotImplementedError(f"metric {metric} not implemented")


def contrastive_loss(
    y, mu_b_c, mu_p_c, logvar_b_c, logvar_p_c, metric="mahalanobis", m=10
):
    """
    y: 0 or 1
    m: margin
    """
    d = divergence_fn(mu_b_c, mu_p_c, logvar_b_c, logvar_p_c, metric)
    loss = (1 - y) * d + y * torch.clamp(m - d, min=0)
    return loss.mean()
