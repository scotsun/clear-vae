"""Elastic Weight Consolidation (EWS)."""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class EWC:
    def __init__(self, model: nn.Module, dataset: Dataset, criterion) -> None:
        self.model = model
        self.dataset = dataset
        self.criterion = criterion

        self.params_old = dict(model.named_parameters())
        self.fisher_information: dict = self._compute_fisher()

    def _compute_fisher(self):
        """Calculate diagonal elements in the information matrix."""
        fisher_information = {}
        for n, p in self.model.named_parameters():
            fisher_information[n] = torch.zeros_like(p)
        N = len(self.dataset)
        for i in range(N):
            self.model.zero_grad()
            x, y = self.dataset[i]
            yh = self.model(x)
            loss = self.criterion(y, yh)
            loss.backward()
            for n, p in self.model.named_parameters():
                fisher_information[n] += p.grad.data**2 / N
        return fisher_information

    def penalty(self, updated_model: nn.Module):
        """Sum over all F(param_new - param_old)^2."""
        loss = 0
        for n, param_new in updated_model.parameters():
            loss += (
                self.fisher_information[n] * (param_new - self.params_old[n]) ** 2
            ).sum()
        return loss
