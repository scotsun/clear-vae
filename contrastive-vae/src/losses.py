"""Different loss & divergence functions."""

import torch
import torch.nn.functional as F
from torch import Tensor, jit
from sklearn.metrics import average_precision_score


def accurary(logit: torch.Tensor, y: torch.Tensor):
    yh = logit.argmax(dim=1)
    return (yh == y).float().mean()


def auc(logit: torch.Tensor, y: torch.Tensor):
    num_classes = y.max() + 1
    ph = logit.softmax(dim=1).detach()
    y_binarized = torch.eye(num_classes)[y]
    auc_scores = dict()
    for i in range(num_classes):
        auc_scores[i] = round(
            average_precision_score(y_binarized[:, i].cpu(), ph[:, i].cpu()), 3
        )
    return auc_scores


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
    return reconstruction_loss, kl_c, kl_s


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


# Pairwise similarity measures
def pairwise_cosine(mu: torch.Tensor):
    return F.cosine_similarity(mu[None, :, :], mu[:, None, :], dim=-1)


def pairwise_variance_adjusted_cosine(mu: torch.Tensor, logvar: torch.Tensor):
    sigma = logvar.exp().sqrt()
    z = mu / sigma
    return F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)


def pairwise_jeffrey_sim(mu: torch.Tensor, logvar: torch.Tensor):
    k = mu.shape[1]
    var = logvar.exp()
    term1 = (var.prod(dim=-1)[None, :] / var.prod(dim=-1)[:, None]).log() - k
    term2 = ((mu[None, :, :] - mu[:, None, :]) ** 2 / var).sum(dim=-1)
    term3 = (var[None, :, :] / var[:, None, :]).sum(dim=-1)

    pairwise_kl = 0.5 * (term1 + term2 + term3)
    pairwise_jeff = pairwise_kl + pairwise_kl.T

    return torch.exp(-pairwise_jeff)


def pairwise_bhattacharyya_coef(mu: torch.Tensor, logvar: torch.Tensor):
    var = logvar.exp()
    var_avg = 0.5 * (var[None, :, :] + var[:, None, :])  # pairwise avg
    term1 = ((mu[None, :, :] - mu[:, None, :]) ** 2 / var_avg).sum(dim=-1)

    # det_var = var.prod(dim=-1)  # get det_sigma_1, det_sigma_2,...
    # term2 = torch.log(
    #     var_avg.prod(dim=-1) / (det_var[None, :] * det_var[:, None]).sqrt()
    # )
    # bd = 1 / 8 * term1 + 1 / 2 * term2
    bc = 2 * torch.exp(-term1) - 1
    return bc


@jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    """Stable logsumexp."""
    # cite: https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim)
    mask = m == -float("inf")

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float("inf"))


# Technically, we are using soft-nearest-neighbor loss (multiple pos pair version of nt-xent)
# but due to SimCLR's popularity, we refer it as nt-xent loss
def _nt_xent_loss(sim: torch.Tensor, pos_target: torch.Tensor, temperature: float):
    n = sim.shape[0]
    sim = sim.clone()
    sim[torch.eye(n).bool()] = float("-Inf")

    neg_mask = pos_target == 0
    pos = pos_target * sim
    pos[neg_mask] = float("-Inf")
    loss = -logsumexp(pos / temperature, dim=1) + logsumexp(sim / temperature, dim=1)
    return loss


def nt_xent_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    label: torch.Tensor,
    sim_fn: str,
    temperature: float,
    flip: bool = False,
):
    if flip:
        pos_target = (label[None, :] != label[:, None]).float()
    else:
        pos_target = (label[None, :] == label[:, None]).float()
    match sim_fn:
        case "cosine":
            sim = pairwise_cosine(mu)
        case "cosine-var-adjust":
            sim = pairwise_variance_adjusted_cosine(mu, logvar)
        case "bhattacharyya-coef":
            sim = pairwise_bhattacharyya_coef(mu, logvar)
        case "jeffrey":
            sim = pairwise_jeffrey_sim(mu, logvar)
        case _:
            raise ValueError("unimplemented similarity measure.")
    losses = _nt_xent_loss(sim, pos_target, temperature)
    finite_mask = torch.isfinite(losses)
    return losses[finite_mask].mean()
