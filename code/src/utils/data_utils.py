"""Data utility functions."""

import numpy as np
import pandas as pd
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
from corruption_utils import corruptions


def random_style_distribution(
    styles: list = [
        corruptions.identity,
        corruptions.stripe,
        corruptions.zigzag,
        corruptions.canny_edges,
    ],
) -> dict:
    probs = np.random.dirichlet([10] * len(styles))
    output = dict()
    for i, fn in enumerate(styles):
        output[fn] = probs[i]
    return output


class StyledMNISTGenerator:
    """A Helper class to fix the random style assignment to each MNIST image."""

    def __init__(
        self, dataset: torchvision.datasets.MNIST, corruption_fns: None | dict
    ) -> None:
        self.dataset = dataset
        self.corruption_fns = list(corruption_fns.keys())
        self.corruption_fns_p = list(corruption_fns.values())

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.corruption_fns is not None:
            cfn_idx = np.random.choice(
                len(self.corruption_fns), p=self.corruption_fns_p
            )
            img = self.corruption_fns[cfn_idx](img)
            return img, label, cfn_idx
        else:
            return img, label, 0

    @property
    def size(self):
        return len(self.dataset)


class StyledMNIST(Dataset):
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

    def display(self, idx):
        img, _, _ = self.__getitem__(idx)
        return transforms.ToPILImage()(img)


ATTR_TO_COLUMN = {
    "blurry": 10,
    "male": 20,
    "smiling": 31,
    "young": -1,
    "black hair": 8,
    "blond hair": 9,
    "brown hair": 11,
    "gray hair": 17,
}

HAIRCOLOR_IDS = [
    ATTR_TO_COLUMN[c] for c in ["black hair", "blond hair", "brown hair", "gray hair"]
]

ATTR_TO_GENDERSMILE_LABEL = {
    # (male, smiling)
    (1, 1): 0,
    (1, 0): 1,
    (0, 1): 2,
    (0, 0): 3,
}

ATTR_TO_HAIRCOLOR_LABEL = {
    "black hair": 1,
    "blond hair": 2,
    "brown hair": 3,
    "gray hair": 4,
}


def generate_celeba_labels(attr):
    gendersmile = ATTR_TO_GENDERSMILE_LABEL[
        (int(attr[ATTR_TO_COLUMN["male"]]), int(attr[ATTR_TO_COLUMN["smiling"]]))
    ]
    hair = attr[HAIRCOLOR_IDS].argmax()
    # argmax will break ties in favor of the first item
    return gendersmile, hair


def get_process_celeba_dataloaders(celeba, splits: list[float]) -> tuple:
    celeba_selected = []
    for img, attr in tqdm(celeba):
        if attr[HAIRCOLOR_IDS].sum() > 0 and attr[ATTR_TO_COLUMN["blurry"]] == 0:
            celeba_selected.append((img, *generate_celeba_labels(attr)))
        else:
            continue

    train, valid, test = random_split(celeba_selected, splits)
    return train, valid, test


class CheXpert(Dataset):
    def __init__(self, data_file: pd.DataFrame) -> None:
        super().__init__()
        self.data_file = data_file
        self.transform = transforms.Compose(
            [
                transforms.Lambda(self._pad_to_square),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self.data_file.shape[0]

    def _pad_to_square(self, img):
        width, height = img.size
        if width == height:
            return img
        # Calculate padding
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2
        right = max_dim - width - left
        bottom = max_dim - height - top
        # Apply padding
        return transforms.functional.pad(img, (left, top, right, bottom), fill=0)

    def __getitem__(self, idx) -> tuple:
        path = "../data/chexpert/" + self.data_file.iloc[idx]["Path"].split("/", 1)[1]
        img = Image.open(path)
        img = self.transform(img)
        return img

    def display(self, idx):
        img = self.__getitem__(idx)
        return transforms.ToPILImage()(img)
