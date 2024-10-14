import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        self.G = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, kernel_size=7, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.G(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.D = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )
        self.dhead = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.dhead(self.D(x))


class WassersteinDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.wD = nn.Sequential(
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
        self.dhead = nn.Linear(2048, 1)

    def forward(self, x):
        return self.dhead(self.wD(x))


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.enc(x)


class DHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dhead = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        return self.dhead(h)


class QHead(nn.Module):
    def __init__(self, c_dim) -> None:
        super().__init__()
        self.c_dim = c_dim
        self.qhead = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, c_dim),
        )

    def forward(self, h):
        return self.qhead(h)
