"""Trainer class"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from src.models.vae import VAE


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


class VAETrainer(Trainer):
    def __init__(
        self,
        vae: nn.Module,
        optimizer: Optimizer,
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(vae, optimizer, verbose_period, device, transform)

    def _valid(self, dataloader, verbose, epoch_id):
        if verbose:
            mig, mse = self.evaluate(dataloader, verbose, epoch_id)
            print(f"gMIG: {round(mig, 3)}; mse: {round(float(mse), 3)}")


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


class HierarchicalVAETrainer(VAETrainer):
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
        eval_evidence_acc: bool = False,
    ):
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch)
            if valid_loader is not None:
                self._valid(valid_loader, verbose, epoch, eval_evidence_acc)

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
            mig, mse = self.evaluate(dataloader, verbose, epoch_id, with_evidence_acc)
            print(f"gMIG: {round(mig, 3)}; mse: {round(float(mse), 3)}")

    def evaluate(self, dataloader, verbose, epoch_id, with_evidence_acc=False):
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
        mse = float(total_reconstr_loss / len(dataloader))

        if verbose:
            print(
                "val_recontr_loss={:.3f}, val_kl_c={:.3f}, val_kl_s={:.3f}".format(
                    total_reconstr_loss / len(dataloader),
                    total_kl_c / len(dataloader),
                    total_kl_s / len(dataloader),
                )
            )

        return mig, mse


class CLEARVAETrainer(VAETrainer):
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
                    + alpha * _ntxent_loss
                    + alpha * _reverse_ntxent_loss
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
                if self.transform:
                    X = self.transform(X)

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
        mse = float(total_reconstr_loss / len(dataloader))

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

        return mig, mse


def factor_shuffling(z: torch.Tensor, strategy: str = "permute_1"):
    """
    z = (z_c, z_s)
    """
    z_dim = int(z.shape[1] / 2)
    z_c, z_s = z[:, :z_dim], z[:, z_dim:]
    match strategy:
        case "full":
            z_s_changed = z_s(torch.randperm(z_s.shape[0]))
            return torch.cat([z_c, z_s_changed], dim=1)
        case "permute_1":
            z_s_changed = torch.cat([z_s[1:, :], z_s[0, :][None]], dim=0)
            return torch.cat([z_c, z_s_changed], dim=1)
        case _:
            raise ValueError("this strategy is not implemented yet")


class ClearTCVAETrainer(VAETrainer):
    def __init__(
        self,
        model: VAE,
        factor_cls: nn.Module,
        optimizers: dict[str, Optimizer],
        sim_fn: str,
        hyperparameter: dict[str, float],
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(
            model, optimizers["vae_optim"], verbose_period, device, transform
        )
        self.sim_fn = sim_fn
        self.factor_optimizer = optimizers["factor_optim"]
        self.factor_cls = factor_cls
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
    ):
        factor_d_losses = []
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch, factor_d_losses)
            if valid_loader is not None:
                self._valid(valid_loader, verbose, epoch)
        return factor_d_losses

    def _train(
        self,
        dataloader: DataLoader,
        verbose: bool,
        epoch_id: int,
        factor_d_losses: list,
    ):
        vae = self.model
        factor_cls = self.factor_cls
        vae.train()
        factor_cls.train()
        vae_optimizer = self.optimizer  # taking enc & dec params
        factor_optimizer = self.factor_optimizer  # taking factor_cls params
        annealer = self.annealer
        device = self.device
        hyperparameter = self.hyperparameter
        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch in bar:
                X, label = batch[0], batch[1].reshape(-1).long()
                X, label = X.to(device), label.to(device)
                if self.transform:
                    X = self.transform(X)

                # vae training
                X_hat, latent_params, z = vae(X, explicit=True)
                vae_optimizer.zero_grad()
                _reconstr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)
                _ntxent_loss = snn_loss(
                    mu=latent_params["mu_c"],
                    logvar=latent_params["logvar_c"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=hyperparameter["temperature"],
                )
                d_score = factor_cls(z)
                _mi_loss = F.relu(torch.log(d_score / (1 - d_score))).mean()

                loss = (
                    _reconstr_loss
                    + annealer(_kl_c)
                    + annealer(_kl_s)
                    + hyperparameter["alpha"] * _ntxent_loss
                    + hyperparameter["lambda"] * _mi_loss
                )

                loss.backward()
                vae_optimizer.step()
                annealer.step()

                # (density-ratio trick) tc factor_cls training
                _, _, z = vae(X, explicit=True)
                z = z.detach()
                factor_optimizer.zero_grad()
                d_score_joint = factor_cls(z)
                d_score_marginals = factor_cls(factor_shuffling(z))
                factor_loss = nn.BCELoss()(
                    torch.cat([d_score_joint, d_score_marginals], dim=0),
                    torch.cat(
                        [
                            torch.ones_like(d_score_joint),
                            torch.zeros_like(d_score_marginals),
                        ],
                        dim=0,
                    ),
                )
                factor_d_losses.append(float(factor_loss))

                # factor cls update
                factor_loss.backward()
                factor_optimizer.step()

                # logging
                bar.set_postfix(
                    factor_cls_loss=float(factor_loss),
                    recontr_loss=float(_reconstr_loss),
                    kl_c=float(_kl_c),
                    kl_s=float(_kl_s),
                    c_loss=float(_ntxent_loss),
                    mi_loss=float(_mi_loss),
                )

    def evaluate(self, dataloader, verbose, epoch_id):
        vae = self.model
        factor_cls = self.factor_cls
        vae.eval()
        factor_cls.eval()
        device = self.device
        hyperparameter = self.hyperparameter
        total_reconstr_loss, total_kl_c, total_kl_s, total_c_loss, total_mi_loss = (
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
                if self.transform:
                    X = self.transform(X)

                X_hat, latent_params, z = vae(X, explicit=True)
                _reconstr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)
                _ntxent_loss = snn_loss(
                    mu=latent_params["mu_c"],
                    logvar=latent_params["logvar_c"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=hyperparameter["temperature"],
                )
                d_score = factor_cls(z)
                _mi_loss = F.relu(torch.log(d_score / (1 - d_score))).mean()

                total_reconstr_loss += _reconstr_loss
                total_kl_c += _kl_c
                total_kl_s += _kl_s
                total_c_loss += _ntxent_loss
                total_mi_loss += _mi_loss

                all_label.append(label)
                all_latent_c.append(z[:, : vae.z_dim])
                all_latent_s.append(z[:, vae.z_dim :])
        all_label, all_latent_c, all_latent_s = (
            torch.cat(all_label),
            torch.cat(all_latent_c),
            torch.cat(all_latent_s),
        )
        mig = mutual_info_gap(all_label, all_latent_c, all_latent_s)
        mse = float(total_reconstr_loss / len(dataloader))

        if verbose:
            print(
                "val_recontr_loss={:.3f}, val_kl_c={:.3f}, val_kl_s={:.3f}, val_c_loss={:.3f}, val_mi_loss={:.3f}".format(
                    total_reconstr_loss / len(dataloader),
                    total_kl_c / len(dataloader),
                    total_kl_s / len(dataloader),
                    total_c_loss / len(dataloader),
                    total_mi_loss / len(dataloader),
                )
            )

        return mig, mse


class ClearMIMVAETrainer(VAETrainer):
    def __init__(
        self,
        model: VAE,
        mi_estimator: nn.Module,
        optimizers: dict[str, Optimizer],
        sim_fn: str,
        hyperparameter: dict[str, float],
        verbose_period: int,
        device: torch.device,
        transform=None,
    ) -> None:
        super().__init__(
            model, optimizers["vae_optim"], verbose_period, device, transform
        )
        self.sim_fn = sim_fn
        self.mi_estimator_optimizer = optimizers["mi_estimator_optim"]
        self.mi_estimator = mi_estimator
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
    ):
        mi_losses, mi_learning_losses = [], []
        for epoch in range(epochs):
            verbose = (epoch % self.verbose_period) == 0
            self._train(train_loader, verbose, epoch, mi_losses, mi_learning_losses)
            if valid_loader is not None:
                self._valid(valid_loader, verbose, epoch)
        return mi_losses, mi_learning_losses

    def _train(
        self,
        dataloader: DataLoader,
        verbose: bool,
        epoch_id: int,
        mi_losses: list,
        mi_learning_losses: list,
    ):
        vae = self.model
        mi_estimator = self.mi_estimator
        vae.train()
        mi_estimator.train()
        vae_optimizer = self.optimizer  # taking enc & dec params
        mi_estimator_optimizer = (
            self.mi_estimator_optimizer
        )  # taking mi_estimator params
        annealer = self.annealer
        device = self.device
        hyperparameter = self.hyperparameter
        with tqdm(dataloader, unit="batch", mininterval=0, disable=not verbose) as bar:
            bar.set_description(f"Epoch {epoch_id}")
            for batch in bar:
                X, label = batch[0], batch[1].reshape(-1).long()
                X, label = X.to(device), label.to(device)
                if self.transform:
                    X = self.transform(X)

                # vae training
                X_hat, latent_params, z = vae(X, explicit=True)
                vae_optimizer.zero_grad()
                _reconstr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)
                _ntxent_loss = snn_loss(
                    mu=latent_params["mu_c"],
                    logvar=latent_params["logvar_c"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=hyperparameter["temperature"],
                )
                _mi_loss = mi_estimator(z[:, : vae.z_dim], z[:, vae.z_dim :])
                mi_losses.append(float(_mi_loss))

                loss = (
                    _reconstr_loss
                    + annealer(_kl_c)
                    + annealer(_kl_s)
                    + hyperparameter["alpha"] * _ntxent_loss
                    + hyperparameter["lambda"] * _mi_loss
                )

                loss.backward()
                vae_optimizer.step()
                annealer.step()

                # learning mi_estimator
                for j in range(5):
                    _, _, z = vae(X, explicit=True)
                    z = z.detach()
                    _mi_learning_loss = mi_estimator.learning_loss(
                        # c -> s
                        z[:, : vae.z_dim],
                        z[:, vae.z_dim :],
                        # s -> c
                        # z[:, vae.z_dim :],
                        # z[:, : vae.z_dim],
                    )
                    mi_estimator_optimizer.zero_grad()
                    _mi_learning_loss.backward()
                    mi_estimator_optimizer.step()
                    mi_learning_losses.append(float(_mi_learning_loss))

                # logging
                bar.set_postfix(
                    recontr_loss=float(_reconstr_loss),
                    kl_c=float(_kl_c),
                    kl_s=float(_kl_s),
                    c_loss=float(_ntxent_loss),
                    mi_loss=float(_mi_loss),
                )

    def evaluate(self, dataloader, verbose, epoch_id):
        vae = self.model
        mi_estimator = self.mi_estimator
        vae.eval()
        mi_estimator.eval()
        device = self.device
        hyperparameter = self.hyperparameter
        total_reconstr_loss, total_kl_c, total_kl_s, total_c_loss, total_mi_loss = (
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
                if self.transform:
                    X = self.transform(X)

                X_hat, latent_params, z = vae(X, explicit=True)
                _reconstr_loss, _kl_c, _kl_s = vae_loss(X_hat, X, **latent_params)
                _ntxent_loss = snn_loss(
                    mu=latent_params["mu_c"],
                    logvar=latent_params["logvar_c"],
                    label=label,
                    sim_fn=self.sim_fn,
                    temperature=hyperparameter["temperature"],
                )
                _mi_loss = mi_estimator(z[:, : vae.z_dim], z[:, vae.z_dim :])

                total_reconstr_loss += _reconstr_loss
                total_kl_c += _kl_c
                total_kl_s += _kl_s
                total_c_loss += _ntxent_loss
                total_mi_loss += _mi_loss

                all_label.append(label)
                all_latent_c.append(z[:, : vae.z_dim])
                all_latent_s.append(z[:, vae.z_dim :])
        all_label, all_latent_c, all_latent_s = (
            torch.cat(all_label),
            torch.cat(all_latent_c),
            torch.cat(all_latent_s),
        )
        mig = mutual_info_gap(all_label, all_latent_c, all_latent_s)
        mse = float(total_reconstr_loss / len(dataloader))

        if verbose:
            print(
                "val_recontr_loss={:.3f}, val_kl_c={:.3f}, val_kl_s={:.3f}, val_c_loss={:.3f}, val_mi_loss={:.3f}".format(
                    total_reconstr_loss / len(dataloader),
                    total_kl_c / len(dataloader),
                    total_kl_s / len(dataloader),
                    total_c_loss / len(dataloader),
                    total_mi_loss / len(dataloader),
                )
            )

        return mig, mse
