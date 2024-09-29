"""Trainer class"""
import math
import numpy as np
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.losses import (
    vae_loss,
    snn_loss,
    auc,
    accurary,
    mutual_info_gap,
)
from src.model import VAE


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
        transform=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.verbose_period = verbose_period
        self.device = device
        self.transform = transform

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


class DownstreamMLPTrainer(Trainer):
    def __init__(
        self,
        vae: nn.Module,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(model, optimizer, verbose_period, device, transform)
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
            for batch in bar:
                X_batch, y_batch = batch[0], batch[1].reshape(-1).long()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if self.transform:
                    X_batch = self.transform(X_batch)
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
            (aupr_scores, auroc_scores), acc = self.evaluate(
                dataloader, verbose, epoch_id
            )
            print("val_aupr:", aupr_scores)
            print(np.mean(list(aupr_scores.values())).round(3))
            print("val_auroc:", auroc_scores)
            print(np.mean(list(auroc_scores.values())).round(3))
            print("val_acc:", acc.numpy().round(3))

    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae = self.vae
        model = self.model
        device = self.device
        model.eval()
        all_y = []
        all_logits = []
        with torch.no_grad():
            for batch in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                X_batch, y_batch = batch[0], batch[1].reshape(-1)
                X_batch = X_batch.to(device)
                mu_c = vae.encode(X_batch)[0]
                logits = model(mu_c)

                all_y.append(y_batch)
                all_logits.append(logits)
        all_y, all_logits = torch.cat(all_y), torch.cat(all_logits)
        return auc(all_logits, all_y), accurary(all_logits, all_y)


class SimpleCNNTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(model, optimizer, verbose_period, device, transform)
        self.criterion = criterion

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        cnn = self.model
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        cnn.train()

        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for batch in bar:
                X_batch, y_batch = batch[0], batch[1].reshape(-1).long()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                if self.transform:
                    X_batch = self.transform(X_batch)
                optimizer.zero_grad()
                logits = cnn(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

                # update running stats
                bar.set_postfix(loss=float(loss))

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        if verbose:
            (aupr_scores, auroc_scores), acc = self.evaluate(
                dataloader, verbose, epoch_id
            )
            print("val_aupr:", aupr_scores)
            print(np.mean(list(aupr_scores.values())).round(3))
            print("val_auroc:", auroc_scores)
            print(np.mean(list(auroc_scores.values())).round(3))
            print("val_acc:", acc.numpy().round(3))

    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        cnn = self.model
        device = self.device
        cnn.eval()

        all_y = []
        all_logits = []
        with torch.no_grad():
            for batch in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                X_batch, y_batch = batch[0], batch[1].reshape(-1)
                X_batch = X_batch.to(device)
                logits = cnn(X_batch)
                all_y.append(y_batch)
                all_logits.append(logits)
        all_y, all_logits = torch.cat(all_y), torch.cat(all_logits)
        return auc(all_logits, all_y), accurary(all_logits, all_y)


class MLVAETrainer(Trainer):
    def __init__(
        self,
        model: VAE,
        optimizer: Optimizer,
        hyperparameter: dict[str, float],
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(model, optimizer, verbose_period, device, transform)
        self.hyperparameter = hyperparameter
        self.annealer = LogisticAnnealer(
            loc=hyperparameter["loc"],
            scale=hyperparameter["scale"],
            beta=hyperparameter["beta"],
        )

    def fit(
        self,
        epochs: int,
        train_loader: DataLoader,
        valid_loader: None | DataLoader = None,
        with_evidence_acc: bool = False,
    ):
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch)
            if valid_loader is not None:
                self._valid(valid_loader, verbose, epoch, with_evidence_acc)

    def _group_adjust(self, B, m, *losses):
        "B: batch size; m: number of groups"
        return [loss * B / m for loss in losses]

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        vae = self.model
        optimizer = self.optimizer
        annealer = self.annealer
        device = self.device
        vae.train()
        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for batch in bar:
                X, label = batch[0], batch[1].reshape(-1).long()
                X, label = X.to(device), label.to(device)
                batch_size, n_groups = X.size(0), len(label.unique())

                if self.transform:
                    X = self.transform(X)
                optimizer.zero_grad()
                X_hat, latent_params = vae(X, label=label)

                _reconstr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)
                _reconstr_loss, _kl_s = self._group_adjust(
                    batch_size, n_groups, _reconstr_loss, _kl_s
                )  # the group-wise loss adjust only applies to train

                loss = _reconstr_loss + annealer(_kl_c) + annealer(_kl_s)

                loss.backward()
                optimizer.step()
                annealer.step()

                bar.set_postfix(
                    reconstr_loss=float(_reconstr_loss),
                    kl_c=float(_kl_c),
                    kl_s=float(_kl_s),
                )

    def _valid(self, dataloader, verbose, epoch_id, with_evidence_acc=False):
        if verbose:
            mig, elbo = self.evaluate(dataloader, verbose, epoch_id, with_evidence_acc)
            print(f"gMIG: {round(mig, 3)}; elbo: {round(float(elbo), 3)}")

    def evaluate(self, dataloader, verbose, epoch_id, with_evidence_acc):
        vae: VAE = self.model
        vae.eval()
        device = self.device
        total_reconstr_loss, total_kl_c, total_kl_s = 0, 0, 0

        all_label = []
        all_latent_c = []
        all_latent_s = []
        with torch.no_grad():
            for batch in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                X, label = batch[0], batch[1].reshape(-1).long()
                X, label = X.to(device), label.to(device)
                if with_evidence_acc:
                    X_hat, latent_params, z = vae(X, label, explicit=True)
                else:
                    X_hat, latent_params, z = vae(X, explicit=True)

                _reconstr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)

                total_reconstr_loss += _reconstr_loss
                total_kl_c += _kl_c
                total_kl_s += _kl_s

                all_label.append(label)
                all_latent_c.append(z[:, : vae.z_dim])
                all_latent_s.append(z[:, vae.z_dim :])
        all_label, all_latent_c, all_latent_s = (
            torch.cat(all_label),
            torch.cat(all_latent_c),
            torch.cat(all_latent_s),
        )
        mig = mutual_info_gap(all_label, all_latent_c, all_latent_s)
        elbo = -float(total_reconstr_loss + total_kl_c + total_kl_s) / len(dataloader)

        if verbose:
            print(
                "val_recontr_loss={:.3f}, val_kl_c={:.3f}, val_kl_s={:.3f}".format(
                    total_reconstr_loss / len(dataloader),
                    total_kl_c / len(dataloader),
                    total_kl_s / len(dataloader),
                )
            )

        return mig, elbo


class CLEARVAETrainer(Trainer):
    def __init__(
        self,
        model: VAE,
        optimizer: Optimizer,
        sim_fn: str,
        hyperparameter: dict[str, float],
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(model, optimizer, verbose_period, device, transform)
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
            for batch in bar:
                X, label = batch[0], batch[1].reshape(-1).long()
                X, label = X.to(device), label.to(device)
                if self.transform:
                    X = self.transform(X)
                optimizer.zero_grad()
                X_hat, latent_params = vae(X)

                _reconstr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)

                _ntxent_loss = snn_loss(
                    mu=latent_params["mu_c"],
                    logvar=latent_params["logvar_c"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=temperature,
                )
                _reverse_ntxent_loss = snn_loss(
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
                    _reconstr_loss
                    + annealer(_kl_c)
                    + annealer(_kl_s)
                    + alpha[0] * _ntxent_loss
                    + alpha[1] * _reverse_ntxent_loss
                )

                loss.backward()
                optimizer.step()
                annealer.step()

                bar.set_postfix(
                    recontr_loss=float(_reconstr_loss),
                    kl_c=float(_kl_c),
                    kl_s=float(_kl_s),
                    c_loss=float(_ntxent_loss),
                    s_loss=float(_reverse_ntxent_loss),
                )
        return

    def _valid(self, dataloader, verbose, epoch_id):
        if verbose:
            mig, elbo = self.evaluate(dataloader, verbose, epoch_id)
            print(f"gMIG: {round(mig, 3)}; elbo: {round(float(elbo), 3)}")

    def evaluate(self, dataloader, verbose, epoch_id):
        vae: VAE = self.model
        vae.eval()
        device = self.device
        temperature = self.hyperparameter["temperature"]
        label_flipping = self.hyperparameter["label_flipping"]
        total_reconstr_loss, total_kl_c, total_kl_s, total_c_loss, total_s_loss = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

        all_label = []
        all_latent_c = []
        all_latent_s = []
        with torch.no_grad():
            for batch in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                X, label = batch[0], batch[1].reshape(-1).long()
                X, label = X.to(device), label.to(device)

                X_hat, latent_params, z = vae(X, explicit=True)

                _recontr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)
                _ntxent_loss = snn_loss(
                    mu=latent_params["mu_c"],
                    logvar=latent_params["logvar_c"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=temperature,
                )
                _reverse_ntxent_loss = snn_loss(
                    mu=latent_params["mu_s"],
                    logvar=latent_params["logvar_s"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=temperature,
                    flip=label_flipping,
                )
                if not label_flipping:
                    _reverse_ntxent_loss = -_reverse_ntxent_loss

                total_reconstr_loss += _recontr_loss
                total_kl_c += _kl_c
                total_kl_s += _kl_s
                total_c_loss += _ntxent_loss
                total_s_loss += _reverse_ntxent_loss

                all_label.append(label)
                all_latent_c.append(z[:, : vae.z_dim])
                all_latent_s.append(z[:, vae.z_dim :])
        all_label, all_latent_c, all_latent_s = (
            torch.cat(all_label),
            torch.cat(all_latent_c),
            torch.cat(all_latent_s),
        )
        mig = mutual_info_gap(all_label, all_latent_c, all_latent_s)
        elbo = -float(total_reconstr_loss + total_kl_c + total_kl_s) / len(dataloader)

        if verbose:
            print(
                "val_recontr_loss={:.3f}, val_kl_c={:.3f}, val_kl_s={:.3f}, val_c_loss={:.3f}, val_s_loss={:.3f}".format(
                    total_reconstr_loss / len(dataloader),
                    total_kl_c / len(dataloader),
                    total_kl_s / len(dataloader),
                    total_c_loss / len(dataloader),
                    total_s_loss / len(dataloader),
                )
            )

        return mig, elbo
