import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torch.optim import Adam
from tqdm import tqdm

from src.model import Generator, QHead, Encoder, DHead
from src.loss import snn_loss


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, torch.nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    return


class GAN:
    def __init__(self, generator, discriminator, lr, verbose_period, device) -> None:
        self.verbose_period = verbose_period
        self.device = device

        self.generator = generator.apply(weights_init).to(device)
        self.disciminator = discriminator.apply(weights_init).to(device)
        # self.loss = nn.BCELoss()
        self.loss = nn.MSELoss()

        self.d_opt = Adam(
            self.disciminator.parameters(), lr=lr["D"], betas=(0.5, 0.999)
        )
        self.g_opt = Adam(self.generator.parameters(), lr=lr["G"], betas=(0.5, 0.999))

    def fit(self, epochs: int, train_loader: DataLoader):
        d_losses, g_losses = [], []
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch, d_losses, g_losses)
        return d_losses, g_losses

    def _train(
        self,
        train_loader: DataLoader,
        verbose: bool,
        epoch: int,
        d_losses: list,
        g_losses: list,
    ):
        with tqdm(train_loader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch}")
            for batch in bar:
                x = batch[0].to(self.device)
                batch_size = x.size(0)

                # 1. Train Discriminator
                self.d_opt.zero_grad()
                z = torch.randn(batch_size, self.generator.z_dim, 1, 1).to(self.device)

                a = torch.ones(batch_size).to(self.device)
                b = torch.zeros(batch_size).to(self.device)

                # compute bce using the real
                real_scores = self.disciminator(x).squeeze()
                d_loss_real = self.loss(real_scores, a)
                # compute bce using the fake
                fake_x = self.generator(z)
                fake_scores = self.disciminator(fake_x.detach()).squeeze()
                d_loss_fake = self.loss(fake_scores, b)

                # opt discriminator
                # max log(D(x)) + log(1 - D(G(z))) <==> min -log(D(x)) - log(1 - D(G(z)))
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
                d_loss.backward()
                self.d_opt.step()
                self.disciminator.eval()

                # 2. Train Generator
                self.g_opt.zero_grad()
                fake_scores = self.disciminator(fake_x).squeeze()
                # min log(1 - D(G(z))) <==> min -log( D(G(z)) )
                g_loss = 0.5 * self.loss(fake_scores, a)
                self.disciminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                self.g_opt.step()

                # update bar
                d_losses.append(float(d_loss))
                g_losses.append(float(g_loss))
                bar.set_postfix(
                    d_loss=d_loss.item(),
                    g_loss=g_loss.item(),
                )

    def generate_x(self, N, device):
        z = torch.randn(N, self.generator.z_dim, 1, 1).to(device)
        xhat = self.generator(z).detach().reshape(N, 1, 28, 28)
        return xhat


def gradient_penalty(D, real_sample, fake_sample, device):
    """Compute the gradient penalty loss for WGAN.

    L2 Regularize/penalize discriminator's weight gradients L2 norm being greater than 1.
    """
    N, C, H, W = real_sample.shape
    alpha = torch.randn((N, 1, 1, 1)).repeat(1, C, H, W).to(device)
    # get X_hat, the interpolation between real samples and fake samples
    interpolated_images = real_sample * alpha + fake_sample * (1 - alpha)
    interpolated_scores = D(interpolated_images)
    # get the grad D(X_hat)
    gradients = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # compute the penalty
    gradients = gradients.reshape(N, -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


class WGAN:
    def __init__(
        self, generator, discriminator, lr, hyperparam: dict, verbose_period, device
    ) -> None:
        self.verbose_period = verbose_period
        self.device = device

        self.hyperparam = hyperparam

        self.generator = generator.apply(weights_init).to(device)
        self.disciminator = discriminator.apply(weights_init).to(device)

        self.d_opt = Adam(
            self.disciminator.parameters(), lr=lr["D"], betas=(0.5, 0.999)
        )
        self.g_opt = Adam(self.generator.parameters(), lr=lr["G"], betas=(0.5, 0.999))

    def fit(self, epochs: int, train_loader: DataLoader):
        d_losses, g_losses = [], []
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch, d_losses, g_losses)
        return d_losses, g_losses

    def _train(
        self,
        train_loader: DataLoader,
        verbose: bool,
        epoch: int,
        d_losses: list,
        g_losses: list,
    ):
        n_critic = self.hyperparam["n_critic"]
        lambda_gp = self.hyperparam["lambda_gp"]

        self.generator.train()
        self.disciminator.train()
        with tqdm(train_loader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch}")
            for batch in bar:
                x = batch[0].to(self.device)
                batch_size = x.size(0)

                # 1. Train Discriminator
                for _ in range(n_critic):
                    self.d_opt.zero_grad()
                    z = torch.randn(batch_size, self.generator.z_dim, 1, 1).to(
                        self.device
                    )

                    # compute using the real
                    real_scores = self.disciminator(x).squeeze()

                    # compute using the fake
                    fake_x = self.generator(z)
                    fake_scores = self.disciminator(fake_x.detach()).squeeze()

                    # compute gp
                    gp = gradient_penalty(
                        D=self.disciminator,
                        real_sample=x,
                        fake_sample=fake_x,
                        device=self.device,
                    )

                    # opt discriminator
                    # max ( D(x) - D(G(z)) ) <==> min - ( D(x) - D(G(z)) ) + gp-regularizer
                    d_loss = -(real_scores.mean() - fake_scores.mean()) + lambda_gp * gp
                    d_loss.backward()
                    self.d_opt.step()

                # 2. Train Generator
                self.g_opt.zero_grad()
                z = torch.randn(batch_size, self.generator.z_dim, 1, 1).to(self.device)
                fake_x = self.generator(z)
                fake_scores = self.disciminator(fake_x).squeeze()
                # min -( D(G(z)) )
                g_loss = -fake_scores.mean()
                self.disciminator.zero_grad()
                self.generator.zero_grad()
                g_loss.backward()
                self.g_opt.step()

                # update bar
                d_losses.append(float(d_loss))
                g_losses.append(float(g_loss))
                bar.set_postfix(
                    d_loss=d_loss.item(),
                    g_loss=g_loss.item(),
                )


class InfoGAN:
    def __init__(
        self, generator, encoder, qhead, dhead, lr, hyperparam, verbose_period, device
    ) -> None:
        self.verbose_period = verbose_period
        self.device = device
        self.hyperparam = hyperparam

        self.generator = generator.apply(weights_init).to(device)
        self.encoder = encoder.apply(weights_init).to(device)
        self.qhead = qhead.to(device)
        self.dhead = dhead.to(device)

        self.z_c_dim = qhead.out_dim
        self.z_s_dim = generator.z_dim - self.z_c_dim

        self.loss = nn.BCELoss()
        # self.qloss = nn.MSELoss()
        self.qloss = nn.CrossEntropyLoss()

        d_params = list(self.encoder.parameters()) + list(self.dhead.parameters())
        g_params = list(self.generator.parameters()) + list(self.qhead.parameters())
        self.d_opt = Adam(d_params, lr=lr, betas=(0.5, 0.999))
        self.g_opt = Adam(g_params, lr=lr, betas=(0.5, 0.999))

    def fit(self, epochs: int, train_loader: DataLoader):
        d_losses, g_losses, q_losses = [], [], []
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch, d_losses, g_losses, q_losses)
        return d_losses, g_losses, q_losses

    def _train(
        self,
        train_loader: DataLoader,
        verbose: bool,
        epoch: int,
        d_losses: list,
        g_losses: list,
        q_losses: list,
    ):
        self.generator.train()
        self.encoder.train()
        self.dhead.train()
        self.qhead.train()

        lambda_info = self.hyperparam["lambda_info"]

        with tqdm(train_loader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch}")
            for batch in bar:
                x = batch[0].to(self.device)
                batch_size = x.size(0)

                # 1. Train Discriminator
                self.d_opt.zero_grad()
                # noise generation
                _category = torch.randint(0, 10, (batch_size,)).to(self.device)
                z_c = nn.functional.one_hot(_category, num_classes=10).view(
                    batch_size, -1, 1, 1
                )
                z_s = torch.randn(batch_size, self.z_s_dim, 1, 1).to(self.device)
                z = torch.cat([z_c, z_s], dim=1)

                real_labels = torch.ones(batch_size).to(self.device)
                fake_labels = torch.zeros(batch_size).to(self.device)

                # compute bce using the real
                real_scores = self.dhead(self.encoder(x)).squeeze()
                d_loss_real = self.loss(real_scores, real_labels)
                # compute bce using the fake
                fake_x = self.generator(z)
                fake_scores = self.dhead(self.encoder(fake_x.detach())).squeeze()
                d_loss_fake = self.loss(fake_scores, fake_labels)

                # opt discriminator
                # max log(D(x)) + log(1 - D(G(z))) <==> min -log(D(x)) - log(1 - D(G(z)))
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_opt.step()

                # 2. Train Generator
                self.g_opt.zero_grad()
                fake_hidden = self.encoder(fake_x)
                fake_scores = self.dhead(fake_hidden).squeeze()
                # min log(1 - D(G(z))) <==> min -log( D(G(z)) )
                g_loss = self.loss(fake_scores, real_labels)
                # min Q(c|fake_x)
                z_c_logits = self.qhead(fake_hidden).squeeze()

                q_loss = self.qloss(z_c_logits, _category)

                (g_loss + lambda_info * q_loss).backward()
                self.g_opt.step()

                # update bar
                d_losses.append(float(d_loss))
                g_losses.append(float(g_loss))
                q_losses.append(float(q_loss))
                bar.set_postfix(
                    d_loss=d_loss.item(),
                    g_loss=g_loss.item(),
                    q_loss=q_loss.item(),
                )


class InfoWGAN:
    def __init__(
        self, generator, encoder, dhead, qhead, lr, hyperparam, verbose_period, device
    ) -> None:
        self.verbose_period = verbose_period
        self.device = device
        self.hyperparam = hyperparam

        self.generator = generator.apply(weights_init).to(device)
        self.encoder = encoder.apply(weights_init).to(device)
        self.qhead = qhead.to(device)
        self.dhead = dhead.to(device)

        self.z_c_dim = qhead.c_dim
        self.z_s_dim = generator.z_dim - self.z_c_dim

        # self.qloss = nn.MSELoss()
        self.qloss = nn.CrossEntropyLoss()

        d_params = list(self.encoder.parameters()) + list(self.dhead.parameters())
        g_params = list(self.generator.parameters()) + list(self.qhead.parameters())
        self.d_opt = Adam(d_params, lr=2 * lr, betas=(0.5, 0.999))
        self.g_opt = Adam(g_params, lr=lr, betas=(0.5, 0.999))

    def fit(self, epochs: int, train_loader: DataLoader):
        d_losses, g_losses, q_losses = [], [], []
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch, d_losses, g_losses, q_losses)
        return d_losses, g_losses, q_losses

    def _train(
        self,
        train_loader: DataLoader,
        verbose: bool,
        epoch: int,
        d_losses: list,
        g_losses: list,
        q_losses: list,
    ):
        self.generator.train()
        self.encoder.train()
        self.dhead.train()
        self.qhead.train()

        n_critic = self.hyperparam["n_critic"]
        lambda_gp = self.hyperparam["lambda_gp"]
        lambda_info = self.hyperparam["lambda_info"]

        with tqdm(train_loader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch}")
            for batch in bar:
                x = batch[0].to(self.device)
                batch_size = x.size(0)

                # 1. Train Discriminator
                for _ in range(n_critic):
                    _category = torch.randint(0, 10, (batch_size,)).to(self.device)
                    z_c = nn.functional.one_hot(_category, num_classes=10).view(
                        batch_size, -1, 1, 1
                    )
                    z_s = torch.randn(batch_size, self.z_s_dim, 1, 1).to(self.device)
                    z = torch.cat([z_c, z_s], dim=1)

                    # compute bce using the real
                    real_scores = self.dhead(self.encoder(x)).squeeze()

                    # compute bce using the fake
                    fake_x = self.generator(z)
                    fake_scores = self.dhead(self.encoder(fake_x)).squeeze()

                    # compute gp
                    gp = gradient_penalty(
                        D=lambda xhat: self.dhead(self.encoder(xhat)),
                        real_sample=x,
                        fake_sample=fake_x,
                        device=self.device,
                    )

                    # opt discriminator
                    # max ( D(x) - D(G(z)) ) <==> min - ( D(x) - D(G(z)) ) + gp-regularizer
                    d_loss = -(real_scores.mean() - fake_scores.mean()) + lambda_gp * gp
                    self.encoder.zero_grad()
                    self.dhead.zero_grad()
                    d_loss.backward(retain_graph=True)
                    self.d_opt.step()

                # 2. Train Generator
                _category = torch.randint(0, 10, (batch_size,)).to(self.device)
                z_c = nn.functional.one_hot(_category, num_classes=10).view(
                    batch_size, -1, 1, 1
                )
                z_s = torch.randn(batch_size, self.z_s_dim, 1, 1).to(self.device)
                z = torch.cat([z_c, z_s], dim=1)

                fake_x = self.generator(z)
                fake_hidden = self.encoder(fake_x)
                fake_scores = self.dhead(fake_hidden).squeeze()
                # min -( D(G(z)) )
                g_loss = -fake_scores.mean()
                # min Q(c|fake_x)
                z_c_hat = self.qhead(fake_hidden)
                q_loss = self.qloss(z_c_hat, _category)

                self.generator.zero_grad()
                self.encoder.zero_grad()
                self.dhead.zero_grad()
                self.qhead.zero_grad()

                (g_loss + lambda_info * q_loss).backward()
                self.g_opt.step()

                # update bar
                d_losses.append(float(d_loss))
                g_losses.append(float(g_loss))
                q_losses.append(float(q_loss))
                bar.set_postfix(
                    d_loss=d_loss.item(),
                    g_loss=g_loss.item(),
                    q_loss=q_loss.item(),
                )


class CLEARInfoGAN:
    def __init__(
        self, z_dim, num_classes, lr, hyperparam, verbose_period, device
    ) -> None:
        self.z_dim = z_dim
        self.z_c_dim, self.z_s_dim = z_dim // 2, z_dim // 2
        self.num_classes = num_classes
        self.verbose_period = verbose_period
        self.device = device
        self.hyperparam = hyperparam

        self.generator = Generator(z_dim).apply(weights_init).to(device)
        self.zc_encoder = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256), nn.ReLU(), nn.Linear(256, self.z_c_dim)
        ).to(device)
        self.zs_encoder = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256), nn.ReLU(), nn.Linear(256, self.z_s_dim)
        ).to(device)
        self.encoder = Encoder().apply(weights_init).to(device)
        self.qhead_c = QHead(out_dim=self.z_c_dim).to(device)
        self.qhead_s = QHead(out_dim=self.z_s_dim).to(device)
        self.dhead = DHead().to(device)

        self.loss = nn.BCELoss()
        self.qloss = nn.MSELoss()
        # self.qloss = nn.CrossEntropyLoss()

        z_params = list(self.zc_encoder.parameters()) + list(
            self.zs_encoder.parameters()
        )
        d_params = list(self.encoder.parameters()) + list(self.dhead.parameters())
        g_params = (
            list(self.generator.parameters())
            + list(self.qhead_c.parameters())
            + list(self.qhead_s.parameters())
        )
        self.z_opt = Adam(z_params, lr=2 * lr)
        self.d_opt = Adam(d_params, lr=lr, betas=(0.5, 0.999))
        self.g_opt = Adam(g_params, lr=lr, betas=(0.5, 0.999))

    def fit(self, epochs: int, train_loader: DataLoader):
        d_losses, g_losses, q_losses = [], [], []
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch, d_losses, g_losses, q_losses)
        return d_losses, g_losses, q_losses

    def _contrastive_step(self, z, label, num_classes, tau):
        self.z_opt.zero_grad()
        label_onehot = nn.functional.one_hot(label, num_classes).to(self.device)
        z_c = self.zc_encoder(torch.cat([z, label_onehot], dim=-1))
        z_s = self.zs_encoder(torch.cat([z, label_onehot], dim=-1))
        loss_c, loss_s = snn_loss(z_c, label, tau), snn_loss(z_s, label, tau, flip=True)
        (loss_c + loss_s).backward()
        self.z_opt.step()
        return z_c.detach(), z_s.detach(), loss_c, loss_s

    def _train(
        self,
        train_loader: DataLoader,
        verbose: bool,
        epoch: int,
        d_losses: list,
        g_losses: list,
        q_losses: list,
    ):
        self.generator.train()
        self.zc_encoder.train()
        self.zs_encoder.train()
        self.encoder.train()
        self.dhead.train()
        self.qhead_c.train()

        lambda_info = self.hyperparam["lambda_info"]
        tau = self.hyperparam["tau"]

        with tqdm(train_loader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch}")
            for batch in bar:
                x = batch[0].to(self.device)
                batch_size = x.size(0)

                # 1. Train Discriminator
                self.d_opt.zero_grad()
                # noise generation
                _category = torch.randint(0, self.num_classes, (batch_size,)).to(
                    self.device
                )
                z_raw = torch.randn(batch_size, self.z_dim).to(self.device)
                z_c, z_s, loss_c, loss_s = self._contrastive_step(
                    z=z_raw, label=_category, num_classes=self.num_classes, tau=tau
                )
                z = torch.cat([z_c, z_s], dim=1)[:, :, None, None]

                real_labels = torch.ones(batch_size).to(self.device)
                fake_labels = torch.zeros(batch_size).to(self.device)

                # compute bce using the real
                real_scores = self.dhead(self.encoder(x)).squeeze()
                d_loss_real = self.loss(real_scores, real_labels)
                # compute bce using the fake
                fake_x = self.generator(z)
                fake_scores = self.dhead(self.encoder(fake_x.detach())).squeeze()
                d_loss_fake = self.loss(fake_scores, fake_labels)

                # opt discriminator
                # max log(D(x)) + log(1 - D(G(z))) <==> min -log(D(x)) - log(1 - D(G(z)))
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_opt.step()

                # 2. Train Generator
                self.g_opt.zero_grad()
                fake_hidden = self.encoder(fake_x)
                fake_scores = self.dhead(fake_hidden).squeeze()
                # min log(1 - D(G(z))) <==> min -log( D(G(z)) )
                g_loss = self.loss(fake_scores, real_labels)
                # min Q(c|fake_x)
                z_c_hat = self.qhead_c(fake_hidden).squeeze()
                z_s_hat = self.qhead_s(fake_hidden).squeeze()

                q_loss = self.qloss(z_c_hat, z_c) + self.qloss(z_s_hat, z_s)

                (g_loss + lambda_info * q_loss).backward()
                self.g_opt.step()

                # update bar
                d_losses.append(float(d_loss))
                g_losses.append(float(g_loss))
                q_losses.append(float(q_loss))
                bar.set_postfix(
                    d_loss=d_loss.item(),
                    g_loss=g_loss.item(),
                    q_loss=q_loss.item(),
                    loss_c=loss_c.item(),
                    loss_s=loss_s.item(),
                )
