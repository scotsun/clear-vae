"""Expr utils"""
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, ConcatDataset


def generate_style_dict(classes: list, styles: list, k: int):
    if k < 1 or k >= len(styles):
        raise ValueError("k must be in [1, len(styles) - 1]")
    style_dict = dict()
    for c in classes:
        train_styles = np.random.choice(styles, k, replace=False)
        test_styles = np.setdiff1d(styles, train_styles)
        style_dict[c] = {"train": train_styles, "test": test_styles}
    return style_dict


class KStyledMNISTGenerator:
    def __init__(
        self, dataset: Dataset, style_fns: list, style_dict: dict, split: str
    ) -> None:
        self.dataset = dataset
        self.style_fns = style_fns
        self.style_dict = style_dict
        self.split = split

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        cfn_idx = np.random.choice(self.style_dict[label][self.split])

        img = self.style_fns[cfn_idx](img)
        return img, label, cfn_idx

    @property
    def size(self):
        return len(self.dataset)


class KStyledMNIST(Dataset):
    def __init__(self, generator, transform) -> None:
        super().__init__()
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

    def __getitem__(self, idx) -> tuple:
        img, label, style = self.dataset[idx]
        img = self.transform(img)
        return img, label, style


class List2Dataset(Dataset):
    def __init__(self, list):
        self.list = list

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return self.list[idx]


def _subset(dataset, content_label: int, style_label: int):
    subset = [x for x in dataset if x[1] == content_label and x[2] == style_label]
    return List2Dataset(subset)


def kceleba_train_test_split(celeba_data, k: int, seed: int):
    classes = list(range(4))
    styles = list(range(4))
    np.random.seed(seed)
    style_dict = generate_style_dict(classes, styles, k)

    train_list, test_list = [], []
    for c in classes:
        train_styles = style_dict[c]["train"]
        for s in train_styles:
            train_list.append(_subset(celeba_data, c, s))
        test_styles = style_dict[c]["test"]
        for s in test_styles:
            test_list.append(_subset(celeba_data, c, s))

    train = ConcatDataset(train_list)
    test = ConcatDataset(test_list)
    return train, test, style_dict
