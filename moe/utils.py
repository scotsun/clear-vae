"""Utility functions."""

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class DummyDataset(Dataset):
    def __init__(self, fpath) -> None:
        super().__init__()
        try:
            data = pd.read_csv(fpath).drop("cluster", axis=1).values
        except KeyError:
            data = pd.read_csv(fpath).values
        self.data = torch.Tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> tuple:
        return self.data[index, 1:], self.data[index, 0]


def visualize_expert_decisions(model, lims, grid_n):
    xs = torch.linspace(start=lims[0], end=lims[1], steps=grid_n)
    ys = torch.linspace(start=lims[0], end=lims[1], steps=grid_n)
    ws = F.softmax(model.gate_mgr(torch.cartesian_prod(ys, xs)), dim=1).detach()

    fig, axs = plt.subplots(1, ws.shape[1], sharey=True)
    for i in range(ws.shape[1]):
        im = axs[i].imshow(
            ws[:, i].reshape(grid_n, grid_n).T,
            extent=[lims[0], lims[1], lims[1], lims[0]],
            vmin=0,
            vmax=1,
        )
        axs[i].invert_yaxis()

    fig.tight_layout(pad=1)
    plt.title("Weight distribution across experts", loc="right")
    fig.colorbar(im, ax=axs, orientation="horizontal")
    plt.show()

    scores = F.sigmoid(model.experts(torch.cartesian_prod(ys, xs))).detach()
    fig, axs = plt.subplots(1, scores.shape[1], sharey=True)
    for i in range(scores.shape[1]):
        im = axs[i].imshow(
            scores[:, i].reshape(grid_n, grid_n).T,
            extent=[lims[0], lims[1], lims[1], lims[0]],
            vmin=0,
            vmax=1,
        )
        axs[i].invert_yaxis()

    fig.tight_layout(pad=1)
    fig.colorbar(im, ax=axs.ravel(), orientation="horizontal")
    plt.title("Score distribution across experts", loc="right")
    plt.show()

    p = model(torch.cartesian_prod(ys, xs)).detach()
    fig, ax = plt.subplots()
    im = ax.imshow(
        p.reshape(grid_n, grid_n).T,
        extent=[lims[0], lims[1], lims[1], lims[0]],
        vmin=0,
        vmax=1,
    )
    ax.invert_yaxis()
    fig.colorbar(im, ax=ax, orientation="horizontal")
    plt.title("Aggregated decision")
    plt.show()
