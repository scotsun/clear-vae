"""Loss functions."""

import torch
import torch.distributions as dis
import torch.nn.functional as F


def kl_estimated_loss(
    z, mu_minor, logvar_minor, mu_major, logvar_major, device: str, reduce: bool
):
    """Calculate KL loss using MC estimation over z."""
    B, m = mu_minor.shape
    N, _ = mu_major.shape

    q = dis.MultivariateNormal(mu_minor, logvar_minor.exp().diag_embed())
    # z.shape is (B, m)
    # define: ratio = prior_pdf / encoder_pdf
    # see details at http://joschu.net/blog/kl-approx.html
    # estimate kl_loss by the sample mean of (ratio - 1) - log_ratio,
    # which ensures positivity, unbiasedness, less variance
    prior = dis.MultivariateNormal(
        mu_major, logvar_major.exp() * torch.eye(m).to(device)
    )
    p_log_prob = prior.log_prob(z[:, None, :].repeat(1, N, 1))
    mg_p_log_prob = p_log_prob.logsumexp(dim=1) - torch.tensor(N).log().to(device)

    log_ratio = mg_p_log_prob - q.log_prob(z)
    # log_ratio.shape = (B,)
    kl_est = (log_ratio.exp() - 1) - log_ratio

    return kl_est.mean() if reduce else kl_est


def kl_ub_loss(mu_minor, logvar_minor, mu_major, logvar_major, reduce: bool):
    """Calculate KL loss's upper bound using another Jensen's Inequality / GM-AM Inequality."""
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

    return kl.mean() if reduce else kl


def kl_lb_loss(
    mu_minor,
    logvar_minor,
    mu_major,
    logvar_major,
    device: str,
    min_k: int,
    reduce: bool,
):
    """Calculate KL loss's lower bound using the 'closest' k majority sample"""
    B, m = mu_minor.shape
    N, _ = mu_major.shape

    _mu_minor = mu_minor[None, :, :].repeat(N, 1, 1)
    _logvar_minor = logvar_minor[None, :, :].repeat(N, 1, 1)
    _mu_major = mu_major[:, None, :].repeat(1, B, 1)
    kl_mat = dis.kl.kl_divergence(
        dis.MultivariateNormal(_mu_minor, torch.diag_embed(_logvar_minor.exp())),
        dis.MultivariateNormal(_mu_major, logvar_major.exp() * torch.eye(m).to(device)),
    )
    _, idx = torch.topk(-kl_mat, k=min_k, dim=0)
    kl_mink = kl_mat.gather(dim=0, index=idx)

    return kl_mink.mean() if reduce else kl_mink.mean(dim=0)


def mgvae_loss(
    x_reconstr,
    x,
    z,
    mu_minor,
    logvar_minor,
    mu_major,
    logvar_major,
    kl_loss_fn: str,
    device: str,
    min_k: int = 256,
    reduce: bool = True,
):
    """
    Calculate the two parts of the loss objective.

    `reduce` (reduction by mean) is set to `True` by default.
    One would set it to `False` for calculating fisher information in EWC.
    """
    n_dims = len(x.shape)
    _reconstruction_loss = F.mse_loss(x_reconstr, x, reduction="none").sum(
        dim=list(range(n_dims))[1:]
    )
    reconstruction_loss = (
        _reconstruction_loss.mean() if reduce else _reconstruction_loss
    )

    match kl_loss_fn:
        case "estimated":
            kl_regularization = kl_estimated_loss(
                z, mu_minor, logvar_minor, mu_major, logvar_major, device, reduce
            )
        case "upper_bound":
            kl_regularization = kl_ub_loss(
                mu_minor, logvar_minor, mu_major, logvar_major, reduce
            )
        case "lower_bound":
            kl_regularization = kl_lb_loss(
                mu_minor, logvar_minor, mu_major, logvar_major, device, min_k, reduce
            )
        case _:
            raise ValueError("incorrect 'kl_loss_fn' value.")
    return reconstruction_loss, kl_regularization
