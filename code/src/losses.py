"""Different loss & divergence functions."""

import torch
import torch.nn.functional as F
from torch import Tensor, jit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif


def mutual_info_gap(label, latent_c, latent_s):
    label, latent_c, latent_s = label.cpu(), latent_c.cpu(), latent_s.cpu()
    p = torch.bincount(label) / len(label)
    H = float(-(p * torch.log(p)).sum())
    mi_c = mutual_info_classif(latent_c, label, discrete_features=False)
    mi_s = mutual_info_classif(latent_s, label, discrete_features=False)
    return (mi_c.mean() - mi_s.mean()) / H


def accurary(logit: torch.Tensor, y: torch.Tensor):
    yh = logit.argmax(dim=1).cpu()
    return (yh.view(-1) == y.view(-1)).float().mean()


def auc(logit: torch.Tensor, y: torch.Tensor):
    num_classes = int(y.max() + 1)
    ph = logit.softmax(dim=1).detach().cpu()
    y = y.cpu()
    y_binarized = torch.eye(num_classes)[y]
    aupr_scores, auroc_scores = dict(), dict()
    for i in range(num_classes):
        aupr_scores[i] = round(average_precision_score(y_binarized[:, i], ph[:, i]), 3)
        auroc_scores[i] = round(roc_auc_score(y_binarized[:, i], ph[:, i]), 3)
    return aupr_scores, auroc_scores


def sample_level_reduction(tensor: Tensor):
    n_dims = len(tensor.shape)
    return tensor.sum(dim=list(range(n_dims))[1:]).mean()


def vae_loss(x_reconstr, x, mu_c, mu_s, logvar_c, logvar_s):
    """
    VAE loss with separating factors.
    """
    reconstruction_loss = sample_level_reduction(
        F.mse_loss(x_reconstr, x, reduction="none")
    )
    kl_c = -0.5 * sample_level_reduction(1 + logvar_c - mu_c.pow(2) - logvar_c.exp())
    kl_s = -0.5 * sample_level_reduction(1 + logvar_s - mu_s.pow(2) - logvar_s.exp())
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


def pairwise_l2(mu: torch.Tensor):
    return -((mu[None, :, :] - mu[:, None, :]) ** 2).sum(dim=-1)


def pairwise_jeffrey_div(mu: torch.Tensor, logvar: torch.Tensor):
    k = mu.shape[1]
    var = logvar.exp()
    term1 = logvar.sum(dim=-1)[None, :] - logvar.sum(dim=-1)[:, None] - k
    term2 = ((mu[None, :, :] - mu[:, None, :]) ** 2 / logvar.exp()).sum(dim=-1)
    term3 = (var[None, :, :] / (var[:, None, :] + 1e-8)).sum(dim=-1)

    pairwise_kl = 0.5 * (term1 + term2 + term3)
    pairwise_jeff = 0.5 * (pairwise_kl + pairwise_kl.T)

    return -pairwise_jeff


def pairwise_mahalanobis_dis(mu: torch.Tensor, logvar: torch.Tensor):
    var = 0.5 * (logvar.exp()[None, :, :] + logvar.exp()[:, None, :])
    dis_mat = ((mu[None, :, :] - mu[:, None, :]) ** 2 / var).sum(dim=-1)
    return -dis_mat


def pairwise_modified_l2_dis(mu: torch.Tensor, logvar: torch.Tensor):
    var = (0.5 * (logvar[None, :, :] + logvar[:, None, :])).exp()
    dis_mat = ((mu[None, :, :] - mu[:, None, :]) ** 2 / var).sum(dim=-1)
    return -dis_mat


@jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    """Stable logsumexp."""
    # cite: https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim)
    mask = m == -float("inf")

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float("inf"))


def _snn_loss(sim: torch.Tensor, pair_mat: torch.Tensor, temperature: float):
    n = sim.shape[0]
    sim = sim.clone()
    sim[torch.eye(n).bool()] = float("-Inf")

    neg_mask = pair_mat == 0
    pos = pair_mat * sim
    pos[neg_mask] = float("-Inf")
    loss = -logsumexp(pos / temperature, dim=1) + logsumexp(sim / temperature, dim=1)
    return loss


def snn_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    label: torch.Tensor,
    sim_fn: str,
    temperature: float,
    flip: bool = False,
):
    if flip:
        pair_mat = (label[None, :] != label[:, None]).float()
    else:
        pair_mat = (label[None, :] == label[:, None]).float()
    match sim_fn:
        case "cosine":
            sim = pairwise_cosine(mu)
        case "l2":
            sim = pairwise_l2(mu)
        case "modified_l2":
            sim = pairwise_modified_l2_dis(mu, logvar)
        case "jeffrey":
            sim = pairwise_jeffrey_div(mu, logvar)
        case "mahalanobis":
            sim = pairwise_mahalanobis_dis(mu, logvar)
        case _:
            raise ValueError("unimplemented similarity measure.")
    losses = _snn_loss(sim, pair_mat, temperature)
    finite_mask = torch.isfinite(losses)
    return losses[finite_mask].mean()


def lam_loss(
    feature_x: torch.Tensor,
    feature_x_tilde: torch.Tensor,
    y: torch.Tensor,
    linear_w: torch.nn.Parameter,
):
    """Labelled LAM."""
    w_y = linear_w[y]

    # Compute contributions: features * weights for y
    contributions_x = feature_x * w_y
    contributions_x_tilde = feature_x_tilde * w_y

    # Mean squared difference
    return ((contributions_x - contributions_x_tilde) ** 2).sum(dim=1).mean()
