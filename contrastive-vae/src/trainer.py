"""Trainer class"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.losses import vae_loss, contrastive_loss, nt_xent_loss, aupr


class LogisticAnnealer:
    def __init__(self, loc, scale, beta) -> None:
        self.current_step = 0
        self.loc = loc
        self.scale = scale
        self.beta = beta

    def __call__(self, kl_loss) -> torch.Tensor:
        return kl_loss * self.slope()

    def slope(self) -> float:
        exponent = -(self.current_step - self.loc) / self.scale
        return self.beta / (1 + math.exp(exponent))

    def step(self) -> None:
        self.current_step += 1
        return


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

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: None | DataLoader = None,
    ):
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch)
            if valid_loader is not None:
                self._valid(valid_loader, verbose, epoch)

    def evaluate(self, **kwarg):
        pass

    def _train(self, **kwarg):
        pass

    def _valid(self, **kwarg):
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

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
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


class DownstreamMLPTrainer(Trainer):
    def __init__(
        self,
        vae: nn.Module,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        verbose_period: int,
        device: torch.device,
    ) -> None:
        super().__init__(model, optimizer, verbose_period, device)
        self.criterion = criterion
        self.vae = vae

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae = self.vae
        model = self.model
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        model.train()

        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for X_batch, y_batch, _ in bar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                mu_c = vae.encode(X_batch)[0]
                logits = model(mu_c)

                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                # update running stats
                bar.set_postfix(loss=float(loss))

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        if verbose:
            aupr_score = self.evaluate(dataloader, verbose, epoch_id)
            print("val_auc:", aupr_score)
            print(np.mean(list(aupr_score.values())).round(3))

    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae = self.vae
        model = self.model
        device = self.device
        model.eval()
        all_y = []
        all_logits = []
        with torch.no_grad():
            for X_batch, y_batch, _ in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                X_batch = X_batch.to(device)
                mu_c = vae.encode(X_batch)[0]
                logits = model(mu_c)

                all_y.append(y_batch)
                all_logits.append(logits)
        all_y, all_logits = torch.cat(all_y), torch.cat(all_logits)
        return aupr(all_logits, all_y)


class SimpleCNNTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        verbose_period: int,
        device: torch.device,
    ) -> None:
        super().__init__(model, optimizer, verbose_period, device)
        self.criterion = criterion

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        cnn = self.model
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        cnn.train()

        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for X_batch, y_batch, _ in bar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = cnn(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                # update running stats
                bar.set_postfix(loss=float(loss))

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        if verbose:
            aupr_score = self.evaluate(dataloader, verbose, epoch_id)
            print("val_auc:", aupr_score)
            print(np.mean(list(aupr_score.values())).round(3))

    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        cnn = self.model
        device = self.device
        cnn.eval()

        all_y = []
        all_logits = []
        with torch.no_grad():
            for X_batch, y_batch, _ in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                X_batch = X_batch.to(device)
                logits = cnn(X_batch)
                all_y.append(y_batch)
                all_logits.append(logits)
        all_y, all_logits = torch.cat(all_y), torch.cat(all_logits)
        return aupr(all_logits, all_y)


class CDVAETrainer(Trainer):
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
        self.annealer = LogisticAnnealer(
            loc=hyperparameter["loc"],
            scale=hyperparameter["scale"],
            beta=hyperparameter["beta"],
        )

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae = self.model
        vae.train()
        optimizer = self.optimizer
        annealer = self.annealer
        device = self.device
        temperature = self.hyperparameter["temperature"]
        alpha = self.hyperparameter["alpha"]
        label_flipping = self.hyperparameter["label_flipping"]
        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for X, label, _ in bar:
                optimizer.zero_grad()
                X = X.to(device)
                label = label.to(device)

                X_hat, latent_params = vae(X)

                _recontr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)

                _kl_c, _kl_s = annealer(_kl_c), annealer(_kl_s)

                _ntxent_loss = nt_xent_loss(
                    mu=latent_params["mu_c"],
                    logvar=latent_params["logvar_c"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=temperature,
                )
                _reverse_ntxent_loss = nt_xent_loss(
                    mu=latent_params["mu_s"],
                    logvar=latent_params["logvar_s"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=temperature,
                    flip=label_flipping,
                )

                if not label_flipping:
                    _reverse_ntxent_loss = -_reverse_ntxent_loss
                loss = (
                    _recontr_loss
                    + _kl_c
                    + _kl_s
                    + alpha[0] * _ntxent_loss
                    + alpha[1] * _reverse_ntxent_loss
                )

                loss.backward()
                optimizer.step()
                annealer.step()

                bar.set_postfix(
                    recontr_loss=float(_recontr_loss),
                    kl_c=float(_kl_c),
                    kl_s=float(_kl_s),
                    c_loss=float(_ntxent_loss),
                    s_loss=float(_reverse_ntxent_loss),
                )
        return

    def _valid(self, dataloader, verbose, epoch_id):
        if verbose:
            vae = self.model
            vae.eval()
            device = self.device
            temperature = self.hyperparameter["temperature"]
            label_flipping = self.hyperparameter["label_flipping"]
            total_recontr_loss, total_kl_c, total_kl_s, total_c_loss, total_s_loss = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )
            with torch.no_grad():
                for X, label, _ in tqdm(
                    dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
                ):
                    X = X.to(device)
                    label = label.to(device)

                    X_hat, latent_params = vae(X)

                    _recontr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)
                    _ntxent_loss = nt_xent_loss(
                        mu=latent_params["mu_c"],
                        logvar=latent_params["logvar_c"],
                        label=label,
                        sim_fn=self.sim_fn,
                        temperature=temperature,
                    )
                    _reverse_ntxent_loss = nt_xent_loss(
                        mu=latent_params["mu_s"],
                        logvar=latent_params["logvar_s"],
                        label=label,
                        sim_fn=self.sim_fn,
                        temperature=temperature,
                        flip=label_flipping,
                    )
                    if not label_flipping:
                        _reverse_ntxent_loss = -_reverse_ntxent_loss

                    total_recontr_loss += _recontr_loss
                    total_kl_c += _kl_c
                    total_kl_s += _kl_s
                    total_c_loss += _ntxent_loss
                    total_s_loss += _reverse_ntxent_loss

            print(
                "val_recontr_loss={:.3f}, val_kl_c={:.3f}, val_kl_s={:.3f}, val_c_loss={:.3f}, val_s_loss={:.3f}".format(
                    total_recontr_loss / len(dataloader),
                    total_kl_c / len(dataloader),
                    total_kl_s / len(dataloader),
                    total_c_loss / len(dataloader),
                    total_s_loss / len(dataloader),
                )
            )
