import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


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
