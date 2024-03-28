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


class Corrputed_Dataset(Dataset):
    def __init__(self, data, label, corruption_fn, transform_fn):
        """
        label is corresponding to the corruption type
        """
        self.data = data
        N = len(self.data)
        self.label = label * torch.ones(N)
        self.corruption_fn = corruption_fn
        self.transform_fn = transform_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.transform_fn(self.corruption_fn(self.data[idx]))
        y = self.label[idx]
        return x, y


def generate_imbalanced_corrupted_dataset(
    orig_data_list, corruption_style_list, transform_fn
) -> Dataset:
    """Create an imbalanced dataset with different corruption approaches."""
    imbalanced_dataset_list = []
    for i, corrupt_fn in enumerate(corruption_style_list):
        data_c = Corrputed_Dataset(orig_data_list[i], i, corrupt_fn, transform_fn)
        imbalanced_dataset_list.append(data_c)
    imbalanced_dataset = ConcatDataset(imbalanced_dataset_list)
    print([len(elem) for elem in orig_data_list])
    return imbalanced_dataset
