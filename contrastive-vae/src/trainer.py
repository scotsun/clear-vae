"""Trainer class"""
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.losses import vae_loss, contrastive_loss, nt_xent_loss


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        verbose_period: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.verbose_period = verbose_period
        self.device = device

    def fit(self, train_loader: DataLoader, epochs: int):
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch)

    def _train(self, **kwarg):
        pass


class CDTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        verbose_period: int,
        fully_supervised: bool,
        device: torch.device,
    ) -> None:
        super().__init__(model, optimizer, verbose_period, device)
        self.fully_supervised = fully_supervised

    def _train(self, dataloader: DataLoader, verbose: int, epoch_id: int):
        self.model.train()
        device = self.device
        vae = self.model
        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for Xb, Xp, label_c, label_s in bar:
                self.optimizer.zero_grad()
                Xb = Xb.to(device)
                Xp = Xp.to(device)
                label_c = label_c.to(device)
                label_s = label_s.to(device)

                Xb_hat, latent_params_b = vae(Xb)
                Xp_hat, latent_params_p = vae(Xp)

                vae_b_loss = vae_loss(Xb_hat, Xb, **latent_params_b)
                vae_p_loss = vae_loss(Xp_hat, Xp, **latent_params_p)
                contrastive_c = contrastive_loss(
                    label_c,
                    latent_params_b["mu_c"],
                    latent_params_p["mu_c"],
                    latent_params_b["logvar_c"],
                    latent_params_p["logvar_c"],
                )

                if self.fully_supervised:  # fully supervised for style match/mismatch
                    contrastive_s = contrastive_loss(
                        label_s,
                        latent_params_b["mu_s"],
                        latent_params_p["mu_s"],
                        latent_params_b["logvar_s"],
                        latent_params_p["logvar_s"],
                        m=20,
                    )
                else:
                    contrastive_s = 0

                loss = vae_b_loss + vae_p_loss + contrastive_c + contrastive_s

                loss.backward()
                self.optimizer.step()

                if self.fully_supervised:
                    bar.set_postfix(
                        vae_b_loss=float(vae_b_loss),
                        vae_p_loss=float(vae_p_loss),
                        contrastive_c=float(contrastive_c),
                        contrastive_s=float(contrastive_s),
                    )
                else:
                    bar.set_postfix(
                        vae_b_loss=float(vae_b_loss),
                        vae_p_loss=float(vae_p_loss),
                        contrastive_c=float(contrastive_c),
                    )
        return


class SimCLRTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        sim_fn: str,
        hyperparameter: dict[str, float],
        verbose_period: int,
        device: torch.device,
    ) -> None:
        super().__init__(model, optimizer, verbose_period, device)
        self.sim_fn = sim_fn
        self.hyperparameter = hyperparameter

    def _train(self, dataloader: DataLoader, verbose: int, epoch_id: int):
        vae = self.model
        optimizer = self.optimizer
        device = self.device
        temperature = self.hyperparameter["temperature"]
        beta = self.hyperparameter["beta"]
        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for X, label in bar:
                optimizer.zero_grad()
                X = X.to(device)
                label = label.to(device)

                X_hat, latent_params = vae(X)

                _vae_loss = vae_loss(X_hat, X, **latent_params)
                ntxent_loss = nt_xent_loss(
                    mu=latent_params["mu_c"],
                    logvar=latent_params["logvar_c"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=temperature,
                )

                loss = _vae_loss + beta * ntxent_loss

                loss.backward()
                optimizer.step()
                bar.set_postfix(
                    vae_loss=float(_vae_loss),
                    c_loss=float(ntxent_loss),
                )
        return
