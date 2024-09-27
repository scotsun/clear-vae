import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE

from src.model import VAE
from src.utils import interpolate_latent


def make_colored_grid(imgs, nrow, color: str):
    grid = make_grid(imgs, nrow=nrow, pad_value=0.25)
    match color:
        case "red":
            grid[0][grid[0] == 0.25] = 1
            grid[1][grid[0] == 0.25] = 0
            grid[2][grid[0] == 0.25] = 0
        case "blue":
            grid[0][grid[2] == 0.25] = 0
            grid[1][grid[2] == 0.25] = 0
            grid[2][grid[2] == 0.25] = 1
        case _:
            raise ValueError("other color not implemented yet")
    return grid


def feature_swapping_plot(z_c, z_s, X, vae: torch.nn.Module):
    with torch.no_grad():
        n = z_c.size(0)
        z_dim = z_c.size(1) + z_s.size(1)

        paired_z = torch.cat(
            (z_c[:, None, :].repeat(1, n, 1), z_s[None, :, :].repeat(n, 1, 1)), dim=-1
        ).view(-1, z_dim)

        hgrid = make_colored_grid(X, nrow=X.size(0), color="blue")
        vgrid = make_colored_grid(X, nrow=1, color="red")

        x_inter = vae.decode(paired_z)
        maingrid = make_grid(x_inter, nrow=X.size(0))

        final_grid = torch.cat(
            [
                torch.cat([torch.ones(3, 32, 32).cuda(), vgrid], dim=1),
                torch.cat([hgrid, maingrid], dim=1),
            ],
            dim=-1,
        )

        plt.imshow(final_grid.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.show()

        return


def interpolation_plot(
    X: torch.Tensor,
    z: torch.Tensor,
    vae: VAE,
    z_dim: int,
    sample_size=10,
    inter_steps=11,
):
    device = X.device
    with torch.no_grad():
        src_ids = torch.randperm(z.size(0))[:sample_size]
        tgt_ids = torch.randperm(z.size(0))[:sample_size]

        X_src, X_tgt = X[src_ids], X[tgt_ids]
        src_grid, tgt_grid = (
            make_colored_grid(X_src, nrow=1, color="red"),
            make_colored_grid(X_tgt, nrow=1, color="blue"),
        )

        _space = torch.ones(src_grid.size(0), src_grid.size(1), 8).to(device)
        src_grid = torch.cat([src_grid, _space], dim=-1)
        tgt_grid = torch.cat([_space, tgt_grid], dim=-1)

        style_inter = []
        content_inter = []
        for i in range(sample_size):
            _src_id, _tgt_id = src_ids[i].item(), tgt_ids[i].item()
            z1, z2 = z[_src_id], z[_tgt_id]
            # style interpolation
            z_inter = interpolate_latent(
                latent1=z1[z_dim:], latent2=z2[z_dim:], num_steps=inter_steps
            )
            z_combined = torch.cat(
                [z1[:z_dim][None, :].repeat(inter_steps, 1), z_inter], dim=1
            )
            style_inter.append(vae.decode(z_combined))

            # content interpolation
            z_inter = interpolate_latent(
                latent1=z1[:z_dim], latent2=z2[:z_dim], num_steps=inter_steps
            )
            z_combined = torch.cat(
                [z_inter, z1[z_dim:][None, :].repeat(inter_steps, 1)], dim=1
            )
            content_inter.append(vae.decode(z_combined))
            # content_inter_grids.append(make_grid(x_inter, nrow=inter_steps))
        style_inter, content_inter = (
            torch.cat(style_inter, dim=0),
            torch.cat(content_inter, dim=0),
        )

        style_inter_grids = make_grid(style_inter, nrow=inter_steps)
        content_inter_grids = make_grid(content_inter, nrow=inter_steps)

        style_inter_grids = torch.cat([src_grid, style_inter_grids, tgt_grid], dim=-1)
        content_inter_grids = torch.cat(
            [src_grid, content_inter_grids, tgt_grid], dim=-1
        )

        plt.imshow(style_inter_grids.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.show()

        plt.imshow(content_inter_grids.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.show()

        return


def _tsne_plot(tsne_2d, class_labels, classes):
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i) for i in np.linspace(0, 1, len(classes))]
    _, ax = plt.subplots()
    for g in range(len(classes)):
        i = np.where(class_labels == g)[0]
        ax.scatter(
            tsne_2d[i, 0], tsne_2d[i, 1], alpha=0.2, c=colors[g], label=classes[g]
        )
    ax.legend()
    plt.show()


def tsne_plot(
    dataloader: DataLoader,
    vae: VAE,
    device: torch.device,
    content_labels=None,
    style_labels=None,
):
    mu_cs, mu_ss, labels, styles = [], [], [], []
    with torch.no_grad():
        for X, c, s in tqdm(dataloader):
            X = X.to(device)
            _, latent_params = vae(X)
            mu_cs.append(latent_params["mu_c"])
            mu_ss.append(latent_params["mu_s"])
            labels.append(c)
            styles.append(s)

    mu_cs, mu_ss, labels, styles = (
        torch.cat(mu_cs, dim=0),
        torch.cat(mu_ss, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(styles, dim=0),
    )

    if content_labels is None:
        content_labels = list(range(labels.max().item() + 1))
    if style_labels is None:
        style_labels = list(range(styles.max().item() + 1))

    tsne_c = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca")
    mu_cs_tsne = tsne_c.fit_transform(mu_cs.cpu().numpy())
    _tsne_plot(mu_cs_tsne, labels.cpu().numpy(), content_labels)
    _tsne_plot(mu_cs_tsne, styles.cpu().numpy(), style_labels)

    tsne_s = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca")
    mu_ss_tsne = tsne_s.fit_transform(mu_ss.cpu().numpy())
    _tsne_plot(mu_ss_tsne, styles.cpu().numpy(), content_labels)
    _tsne_plot(mu_ss_tsne, labels.cpu().numpy(), style_labels)

    return
