"""Data utility functions."""

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from tqdm import tqdm
from IPython import display
import torchvision.transforms as transforms


class MNISTPairGenerator:
    def __init__(
        self, dataset: torchvision.datasets.MNIST, p: float, corruption_fns: None | dict
    ) -> None:
        label_dict = dict()
        for i in range(10):
            label_dict[i] = torch.where(dataset.train_labels == i)[0]
        self.dataset = dataset
        self.corruption_fns = corruption_fns
        self.p = p
        self.label_dict = label_dict

    def __getitem__(self, index):
        base_img, target = self.dataset[index]
        pair_content_label = np.random.binomial(1, self.p)
        if pair_content_label == 0:  # matched
            pair_img_idx = np.random.choice(self.label_dict[target], size=1)[0]
        else:  # unmatched
            unmatched_idx = torch.where(self.dataset.train_labels != target)[0]
            pair_img_idx = np.random.choice(unmatched_idx, size=1)[0]

        if self.corruption_fns is not None:
            base_cfn = np.random.choice(
                list(self.corruption_fns.keys()), p=list(self.corruption_fns.values())
            )
            pair_cfn = np.random.choice(
                list(self.corruption_fns.keys()), p=list(self.corruption_fns.values())
            )

            pair_style_label = 0 if base_cfn == pair_cfn else 1
        else:
            pair_style_label = 0

        pair_img, _ = self.dataset[pair_img_idx]
        return (
            base_cfn(base_img),
            pair_cfn(pair_img),
            pair_content_label,
            pair_style_label,
        )

    @property
    def size(self):
        return len(self.dataset)


class PairDataset(Dataset):
    """
    Dataset object based on a pair generator.
    """

    def __init__(self, generator, transform):
        self.generator = generator
        self.transform = transform
        self.N = generator.size
        self.dataset = [None] * self.N
        with tqdm(range(self.N), unit="item") as bar:
            bar.set_description("Generating dataset")
            for i in bar:
                self.dataset[i] = self.generator[i]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img1, img2, content_label, style_label = self.dataset[idx]
        img1, img2 = self.transform(img1), self.transform(img2)
        return img1, img2, content_label, style_label

    def display(self, idx):
        img1, img2, content_label, style_label = self.__getitem__(idx)
        print(f"content label: {int(content_label)}, style label: {style_label}")
        display(transforms.ToPILImage()(img1), transforms.ToPILImage()(img2))
