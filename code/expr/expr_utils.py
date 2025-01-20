from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


def generate_style_dict(classes: list, style_fns: list, k: int):
    style_dict = dict()
    for c in classes:
        train_fns = np.random.choice(len(style_fns), k, replace=False)
        test_fns = np.setdiff1d(np.arange(len(style_fns)), train_fns)
        style_dict[c] = {
            "train": [style_fns[i] for i in train_fns],
            "test": [style_fns[i] for i in test_fns],
        }
    return style_dict


class KStyledMNISTGenerator:
    def __init__(self, dataset: Dataset, style_dict: dict, split: str) -> None:
        self.dataset = dataset
        self.style_dict = style_dict
        self.split = split

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        cfns = self.style_dict[label][self.split]
        cfn_idx = np.random.choice(len(cfns))

        img = cfns[cfn_idx](img)
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
