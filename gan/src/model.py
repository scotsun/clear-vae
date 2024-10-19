import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.G = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 128, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.G(z)


class Discriminator(nn.Module):
    def __init__(self, gan_type: str = "least-square"):
        super().__init__()
        self.gan_type = gan_type
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 1024, kernel_size=7, stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )
        self.dhead = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        match self.gan_type:
            case "vanilla":
                return torch.sigmoid(self.dhead(self.enc(x)))
            case "least-square":
                return self.dhead(self.enc(x))
            case "wasserstein":
                return self.dhead(self.enc(x))
            case _:
                raise NotImplementedError(f"{self.gan_type} is not implemented")


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 1024, kernel_size=7, stride=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        return self.enc(x)


class WassersteinEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.enc(x)


class DHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dhead = nn.Conv2d(1024, 1, 1)

    def forward(self, h):
        return torch.sigmoid(self.dhead(h))


class QHead(nn.Module):
    def __init__(self, out_dim) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, out_dim, 1),
        )

    def forward(self, h):
        return self.conv(h)
