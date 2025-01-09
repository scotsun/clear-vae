import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch import Tensor, jit
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score


def accurary(logit: torch.Tensor, y: torch.Tensor):
    yh = logit.argmax(dim=1)
    return float((yh.view(-1) == y.view(-1)).float().mean())


def auc(logit: torch.Tensor, y: torch.Tensor):
    num_classes = int(y.max() + 1)
    ph = logit.softmax(dim=1).detach().cpu()
    y = y.cpu()
    y_binarized = torch.eye(num_classes)[y]
    aupr_scores, auroc_scores = dict(), dict()
    for i in range(num_classes):
        aupr_scores[i] = round(average_precision_score(y_binarized[:, i], ph[:, i]), 3)
        auroc_scores[i] = round(roc_auc_score(y_binarized[:, i], ph[:, i]), 3)
    return aupr_scores, auroc_scores


# Pairwise similarity measures
def pairwise_cosine(z: torch.Tensor):
    return F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)


def pairwise_l2(z: torch.Tensor):
    return -((z[None, :, :] - z[:, None, :]) ** 2).sum(dim=-1)


@jit.script
def logsumexp(x: Tensor, dim: int) -> Tensor:
    """Stable logsumexp."""
    # cite: https://github.com/pytorch/pytorch/issues/31829
    m, _ = x.max(dim=dim)
    mask = m == -float("inf")

    s = (x - m.masked_fill_(mask, 0).unsqueeze(dim=dim)).exp().sum(dim=dim)
    return s.masked_fill_(mask, 1).log() + m.masked_fill_(mask, -float("inf"))


def _snn_loss(sim: torch.Tensor, pair_mat: torch.Tensor, temperature: float):
    n = sim.shape[0]
    sim = sim.clone()
    sim[torch.eye(n).bool()] = float("-Inf")

    neg_mask = pair_mat == 0
    pos = pair_mat * sim
    pos[neg_mask] = float("-Inf")
    loss = -logsumexp(pos / temperature, dim=1) + logsumexp(sim / temperature, dim=1)
    return loss


def snn_loss(
    z: torch.Tensor,
    label: torch.Tensor,
    temperature: float,
    flip: bool = False,
):
    if not flip:
        pair_mat = (label[None, :] == label[:, None]).float()  # pair matrix
    else:
        pair_mat = (label[None, :] != label[:, None]).float()
    sim = pairwise_l2(z)  # similarity matrix
    losses = _snn_loss(sim, pair_mat, temperature)
    finite_mask = torch.isfinite(losses)
    return losses[finite_mask].mean()


class MLPTrainer:
    def __init__(
        self,
        mlp: nn.Module,
        cls_head: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        verbose_period: int,
        device: torch.device,
        pair_switch: bool,
    ) -> None:
        self.mlp = mlp
        self.cls_head = cls_head
        self.ps = pair_switch
        self.optimizer = optimizer
        self.verbose_period = verbose_period
        self.device = device
        self.criterion = criterion

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

    def _train(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        device = self.device
        optimizer = self.optimizer
        criterion = self.criterion
        self.mlp.train()
        self.cls_head.train()

        with tqdm(dataloader, unit="batch", disable=not verbose) as bar:
            bar.set_description(f"epoch {epoch_id}")
            for batch in bar:
                X_batch, y_batch = batch[0].view(-1, 784), batch[1].view(-1).long()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                h = self.mlp(X_batch)
                logits = self.cls_head(h)

                if self.ps:
                    # stylistic label
                    label_batch = batch[-1].long().to(device)
                    ps_loss = snn_loss(h, label_batch, temperature=0.07, flip=True)

                    loss = criterion(logits, y_batch)
                    (loss + 10 * ps_loss).backward()
                    optimizer.step()
                    bar.set_postfix(
                        loss=float(loss),
                        ps_loss=float(ps_loss),
                    )
                else:
                    loss = criterion(logits, y_batch)
                    loss.backward()
                    optimizer.step()
                    bar.set_postfix(loss=float(loss))

    def _valid(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        if verbose:
            (aupr_score, auroc_score), acc = self.evaluate(
                dataloader, verbose, epoch_id
            )
            print(f"val_acc: {round(acc, 3)}")
            # print(f"val_aupr: {aupr_score}")
            # print(f"val_auroc: {auroc_score}")

    def evaluate(self, dataloader: DataLoader, verbose: bool, epoch_id: int):
        device = self.device
        self.mlp.eval()
        self.cls_head.eval()
        all_y = []
        all_logits = []
        with torch.no_grad():
            for batch in tqdm(
                dataloader, disable=not verbose, desc=f"val-epoch {epoch_id}"
            ):
                X_batch, y_batch = batch[0].view(-1, 784), batch[1].view(-1).long()
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                h = self.mlp(X_batch)
                logits = self.cls_head(h)

                all_y.append(y_batch)
                all_logits.append(logits)
        all_y, all_logits = torch.cat(all_y), torch.cat(all_logits)
        return auc(all_logits, all_y), accurary(all_logits, all_y)
