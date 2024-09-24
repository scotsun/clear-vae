"""Data utility functions."""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


from corruption_utils import corruptions
from src.model import VAE


def interpolate_latent(latent1, latent2, num_steps, device):
    """Interpolate between two latent vectors."""
    p = torch.linspace(1, 0, num_steps).to(device)
    # reshape to interpolation matrix shape (num_step, latent_dim)
    latent_dim = latent1.shape[-1]
    p = p[:, None].repeat((1, latent_dim))
    latent1 = latent1[None, :].repeat((num_steps, 1))
    latent2 = latent2[None, :].repeat((num_steps, 1))
    # generate interpolation matrix
    return p * latent1 + (1 - p) * latent2


def display_util(idx1, idx2, z: torch.Tensor, model: VAE, z_dim, device):
    with torch.no_grad():
        z1, z2 = z[idx1], z[idx2]

        img1 = transforms.ToPILImage()(model.decode(z1.view(1, -1))[0])
        img2 = transforms.ToPILImage()(model.decode(z2.view(1, -1))[0])

        z_inter = interpolate_latent(
            latent1=z1[z_dim:], latent2=z2[z_dim:], num_steps=11, device=device
        )
        z_combined = torch.cat([z1[:z_dim][None, :].repeat(11, 1), z_inter], dim=1)
        x_inter = model.decode(z_combined)
        print("interpolate style:")
        plt.imshow(make_grid(x_inter, nrow=11).permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.show()

        z_inter = interpolate_latent(
            latent1=z1[:z_dim], latent2=z2[:z_dim], num_steps=11, device=device
        )
        z_combined = torch.cat([z_inter, z1[z_dim:][None, :].repeat(11, 1)], dim=1)
        x_inter = model.decode(z_combined)
        print("interpolate content:")
        plt.imshow(make_grid(x_inter, nrow=11).permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.show()

        return img1, img2


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
        return transforms.ToPILImage()(img1), transforms.ToPILImage()(img2)


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
    def __init__(self, dataset: Dataset, corruption_fns: None | dict) -> None:
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
    "male": 20,
    "smiling": 31,
    "bold": 4,
    "black hair": 8,
    "blond hair": 9,
    "brown hair": 11,
    "gray hair": 17,
}

HAIRCOLOR_IDS = [
    ATTR_TO_COLUMN[c]
    for c in ["bold", "black hair", "blond hair", "brown hair", "gray hair"]
]

ATTR_TO_CONTENT_LABEL = {
    # (male, smiling): content label
    (1, 1): 0,
    (1, 0): 1,
    (0, 1): 2,
    (0, 0): 3,
}

ATTR_TO_STYLE_LABEL = {
    "bold": 0,
    "black hair": 1,
    "blond hair": 2,
    "brown hair": 3,
    "gray hair": 4,
}


def generate_celeba_labels(attr: torch.Tensor):
    # get content label
    content = ATTR_TO_CONTENT_LABEL[
        (int(attr[ATTR_TO_COLUMN["male"]]), int(attr[ATTR_TO_COLUMN["smiling"]]))
    ]
    # get style label
    style = attr[HAIRCOLOR_IDS].argmax()
    # argmax will break ties in favor of the first item
    # so if a person is bold and has other color, he/she will be labeled as bold

    return content, style
