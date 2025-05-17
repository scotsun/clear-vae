# Contrastive LEarning with Anti-contrastive Regularization (CLEAR)
 
This repository contains the code for the paper: CLEAR


## Requirements
- python >= 3.10.4  
- torch >= 2.2.1+cu121  
- sklearn >= 1.4.2
- wandb == 0.6.13

### [MNIST-C](https://github.com/google-research/mnist-c) module `corrpution_utils`

Check https://docs.wand-py.org/en/0.6.7/guide/install.html for installation of `ImageMagick`.
For Ubuntu operating system (e.g. `Google Colab`), try this:
```
apt-get install libmagickwand-dev
pip install wand
```

## Dataset

- Styled- and Colored-MNIST are augmented version of MNIST downloaded through `torchvision` use the code module `corrpution_utils/corruption`.
- CelebA is downloaded through `torchvision`.
- PACS is downloaded through HuggingFace's package `datasets` with name `flwrlabs/pacs`.
- Camelyon17 is downloaded through WILDS benchmark package `wilds`
