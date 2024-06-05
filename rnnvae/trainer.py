"""Trainer class"""
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm


class LogisticAnnealer:
    def __init__(self, loc, scale) -> None:
        self.current_step = 0
        self.loc = loc
        self.scale = scale

    def __call__(self, kl_loss) -> torch.Tensor:
        return kl_loss * self.slope()

    def slope(self) -> float:
        exponent = -(self.current_step - self.loc) / self.scale
        return 1 / (1 + math.exp(exponent))

    def step(self) -> None:
        self.current_step += 1
        return


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        verbose_period: int,
        device: torch.device,
    ) -> None:
        self.model = model
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
            torch.cuda.empty_cache()
            if valid_loader is not None:
                self._valid(valid_loader, verbose, epoch)
            torch.cuda.empty_cache()

    def _train(self, **kwarg):
        pass

    def _valid(self, **kwarg):
        pass


class RVAETrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        verbose_period: int,
        device: torch.device,
        params: dict,
    ) -> None:
        super().__init__(model, verbose_period, device)
        self.optimizer = optimizer
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
        self.annealer = LogisticAnnealer(loc=params["loc"], scale=params["scale"])

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id):
        model = self.model
        device = self.device
        optimizer = self.optimizer
        annealer = self.annealer
        model.train()

        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for _, encoder_inputs in bar:
                encoder_inputs = encoder_inputs.to(device)

                optimizer.zero_grad()

                logits, mu, logvar = model(encoder_inputs)

                reconstr_loss = self.cross_entropy(
                    logits.view(-1, logits.size(-1)), encoder_inputs.view(-1)
                )

                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                _kl_loss = annealer(kl_loss)
                loss = reconstr_loss + _kl_loss

                loss.backward()
                optimizer.step()
                annealer.step()

                # update running stats
                bar.set_postfix(
                    reconstr_loss=float(reconstr_loss), kl_loss=float(kl_loss)
                )

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        model = self.model
        device = self.device
        model.eval()

        total_reconstr_loss, total_kl_loss = 0.0, 0.0
        with torch.no_grad():
            for _, encoder_inputs in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                encoder_inputs = encoder_inputs.to(device)

                logits, mu, logvar = model(encoder_inputs)

                reconstr_loss = self.cross_entropy(
                    logits.view(-1, logits.size(-1)), encoder_inputs.view(-1)
                )
                kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

                total_reconstr_loss += reconstr_loss
                total_kl_loss += kl_loss

            if verbose:
                print(
                    "val_recontr_loss={:.3f}, val_kl_loss={:.3f}".format(
                        total_reconstr_loss / len(dataloader),
                        total_kl_loss / len(dataloader),
                    )
                )
