import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from tqdm import tqdm

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
