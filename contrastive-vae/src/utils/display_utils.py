"""Display utility functions."""

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.models.vae import VAE


def interpolate_latent(latent1, latent2, num_steps):
    """Interpolate between two latent vectors."""
    device = latent1.device
    p = torch.linspace(1, 0, num_steps).to(device)
    # reshape to interpolation matrix shape (num_step, latent_dim)
    latent_dim = latent1.shape[-1]
    p = p[:, None].repeat((1, latent_dim))
    latent1 = latent1[None, :].repeat((num_steps, 1))
    latent2 = latent2[None, :].repeat((num_steps, 1))
    # generate interpolation matrix
    return p * latent1 + (1 - p) * latent2


def display_util(idx1, idx2, z: torch.Tensor, model: VAE, z_dim, device):
    with torch.no_grad():
        z1, z2 = z[idx1], z[idx2]

        img1 = transforms.ToPILImage()(model.decode(z1.view(1, -1))[0])
        img2 = transforms.ToPILImage()(model.decode(z2.view(1, -1))[0])

        z_inter = interpolate_latent(
            latent1=z1[z_dim:], latent2=z2[z_dim:], num_steps=11
        )
        z_combined = torch.cat([z1[:z_dim][None, :].repeat(11, 1), z_inter], dim=1)
        x_inter = model.decode(z_combined)
        print("interpolate style:")
        plt.imshow(make_grid(x_inter, nrow=11).permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.show()

        z_inter = interpolate_latent(
            latent1=z1[:z_dim], latent2=z2[:z_dim], num_steps=11
        )
        z_combined = torch.cat([z_inter, z1[z_dim:][None, :].repeat(11, 1)], dim=1)
        x_inter = model.decode(z_combined)
        print("interpolate content:")
        plt.imshow(make_grid(x_inter, nrow=11).permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.show()

        return img1, img2
