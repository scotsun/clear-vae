"""Data utility functions."""

import random
from torch.utils.data import Dataset
from tqdm import tqdm
from IPython import display
import torchvision.transforms as transforms


class ImbalancePairGenerator:
    """
    Object that make pairs randomly from a imbalanced dataset.
    """

    def __init__(
        self,
        majority_data: list,
        minority_data: list,
        corruption_fn_choices: None | list = None,
    ) -> None:
        self.majority_data = majority_data
        self.minority_data = minority_data
        self.corruption_fn_choices = corruption_fn_choices

    def _sample_img(self, from_majority: bool):
        """helper function to sample a pairing img."""
        if from_majority:
            img, _ = random.choice(self.majority_data)
        else:
            img, _ = random.choice(self.minority_data)
        return img

    def __getitem__(self, idx):
        if idx < len(self.majority_data):
            base_img, _ = self.majority_data[idx]
            pair_img = self._sample_img(from_majority=True)
            label = 0.0
        elif idx >= len(self.majority_data) and idx < len(self.majority_data) + len(
            self.minority_data
        ):
            base_img, _ = self.minority_data[idx - len(self.majority_data)]
            pair_img = self._sample_img(from_majority=True)
            label = 1.0
        else:
            raise IndexError("Index out of range")
        # add corruption
        if self.corruption_fn_choices:
            corruption_fn_base = random.choice(self.corruption_fn_choices)
            corruption_fn_pair = random.choice(self.corruption_fn_choices)
            base_img = corruption_fn_base(base_img)
            pair_img = corruption_fn_pair(pair_img)
        return base_img, pair_img, label


class PairDataset(Dataset):
    """
    Dataset object based on a pair generator.
    """

    def __init__(self, generator: ImbalancePairGenerator, transform):
        self.generator = generator
        self.transform = transform
        self.N = len(generator.majority_data) + len(generator.minority_data)
        self.dataset = [None] * self.N
        with tqdm(range(self.N), unit="item") as bar:
            bar.set_description("Generating dataset")
            for i in bar:
                self.dataset[i] = self.generator[i]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img1, img2, label = self.dataset[idx]
        img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, label

    def display(self, idx):
        img1, img2, pair_label = self.__getitem__(idx)
        print(f"label: {int(pair_label)}")
        display(transforms.ToPILImage()(img1), transforms.ToPILImage()(img2))
