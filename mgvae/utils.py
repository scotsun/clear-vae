"""Utility functions for preparing imbalanced dataset."""

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset, ConcatDataset, Subset


def normalize(x: torch.Tensor):
    """Normalize pixel values to the range of 0-1."""
    return x / 255


def generate_imbalanced_dataset(
    dataset: Dataset, rho: float, transform_fn, rule: int = 5
) -> Dataset:
    """Create an imbalanced version of a standard dataset e.g. MNIST."""
    majority_idx = dataset.targets < rule
    majority_size = majority_idx.sum().item()
    minority_size = (~majority_idx).sum().item()
    majority = TensorDataset(
        transform_fn(dataset.data[majority_idx]), torch.zeros(majority_size)
    )
    minority = TensorDataset(
        transform_fn(dataset.data[~majority_idx]), torch.ones(minority_size)
    )

    if majority_size / minority_size < rho:
        minority = Subset(
            minority,
            np.random.choice(minority_size, int(majority_size / rho), replace=False),
        )

    imbalanced_dataset = ConcatDataset([majority, minority])
    print(f"majority size: {len(majority)}, minority size: {len(minority)}")
    return imbalanced_dataset
