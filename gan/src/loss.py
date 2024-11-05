import torch
import torch.nn.functional as F
from torch import Tensor, jit


# Pairwise similarity measures
def pairwise_cosine(mu: torch.Tensor):
    return F.cosine_similarity(mu[None, :, :], mu[:, None, :], dim=-1)


def pairwise_l2(mu: torch.Tensor):
    return -((mu[None, :, :] - mu[:, None, :]) ** 2).sum(dim=-1)


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
    z: torch.Tensor,
    label: torch.Tensor,
    temperature: float,
    flip: bool = False,
):
    if not flip:
        pair_mat = (label[None, :] == label[:, None]).float()  # pair matrix
    else:
        pair_mat = (label[None, :] != label[:, None]).float()
    sim = pairwise_cosine(z)  # similarity matrix
    losses = _snn_loss(sim, pair_mat, temperature)
    finite_mask = torch.isfinite(losses)
    return losses[finite_mask].mean()
