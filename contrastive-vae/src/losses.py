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


# Similarity measures
def pairwise_cosine(mu: torch.Tensor):
    return F.cosine_similarity(mu[None, :, :], mu[:, None, :], dim=-1)


def pairwise_variance_adjusted_cosine(mu: torch.Tensor, logvar: torch.Tensor):
    sigma = logvar.exp().sqrt()
    z = mu / sigma
    return F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)


def pairwise_bhattacharyya_coef(mu: torch.Tensor, logvar: torch.Tensor):
    sigma = logvar.exp().sqrt()
    sigma_avg = 0.5 * (sigma[None, :, :] + sigma[:, None, :])  # pairwise avg
    term1 = ((mu[None, :, :] - mu[:, None, :]) ** 2 / sigma_avg).sum(dim=-1)

    det_sigma = sigma.prod(dim=-1)  # get det_sigma_1, det_sigma_2,...
    term2 = torch.log(
        sigma_avg.prod(dim=-1) / (det_sigma[None, :] * det_sigma[:, None]).sqrt()
    )
    # pairwise
    bd = 1 / 8 * term1 + 1 / 2 * term2
    bc = torch.exp(-bd)
    return bc


# NT-Xent Loss
def _nt_xent_loss(sim: torch.Tensor, pos_target: torch.Tensor):
    n = sim.shape[0]
    sim[torch.eye(n).bool()] = float("-Inf")

    neg_mask = pos_target == 0
    pos = pos_target * sim
    pos[neg_mask] = float("-Inf")

    loss = -pos.logsumexp(dim=1) + sim.logsumexp(dim=1)
    return loss


def nt_xent_loss(
    mu: torch.Tensor, logvar: torch.Tensor, pos_target: torch.Tensor, sim_fn
):
    match sim_fn:
        case "cosine":
            sim = pairwise_cosine(mu)
        case "cosine-var-adjust":
            sim = pairwise_variance_adjusted_cosine(mu, logvar)
        case "bhattacharyya-coef":
            sim = pairwise_bhattacharyya_coef(mu, logvar)
        case _:
            raise ValueError("unimplemented similarity measure.")
    return _nt_xent_loss(sim, pos_target)
