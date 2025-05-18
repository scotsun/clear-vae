# CLEAR: Unlearning Spurious Style-Content Associations with Contrastive Learning and Anti-contrastive Regularization
 
This repository contains the code for the paper *CLEAR: Unlearning Spurious Style-Content Associations with Contrastive Learning and Anti-contrastive Regularization*


## Requirements
- python >= 3.10.4  
- torch >= 2.2.1+cu121  
- sklearn >= 1.4.2
- wandb == 0.6.13
- datasets 
- wilds

### [MNIST-C](https://github.com/google-research/mnist-c) module `corrpution_utils`

Check https://docs.wand-py.org/en/0.6.7/guide/install.html for installation of `ImageMagick`.
For Ubuntu operating system (e.g. `Google Colab`), try this:
```
apt-get install libmagickwand-dev
pip install wand
```

### Dataset

- Styled- and Colored-MNIST are augmented version of MNIST downloaded through `torchvision` use the code module `corrpution_utils/corruption`.
- CelebA is downloaded through `torchvision`.
- PACS is downloaded through HuggingFace's package `datasets` with name `flwrlabs/pacs`.
- Camelyon17 is downloaded through WILDS benchmark package `wilds`

## Train

The demo files provides the example of quick implementations of different methods on Styled-MNIST, where `demo_clearmimvae.ipynb` is for both L1OutUB and CLUB-S variants.

## Experiments

- `mi_experiment.ipynb` provides the simulation study on the effectiveness of MI max/minimization.
- `swapping_interpolation.ipynb` provides examplar code for performing swapping and interpolation.
- `run*.ipynb` execute the `run*.py` (in `Google Colab` computing environment) for the train-test style shift OOD classification experiments.
    - `run_camelyon17_downstream_expr.ipynb` is following the original experimental protocal, so we don't have a special `.py` file to prepare the data for it.

## Code Details

### `src` Folder
The `src` folder contains the source code for the CLEAR project. This includes:
- Implementation of the different models, metrics/losses, training frameworks.
- Utility functions to support data preprocessing, data visualizing, trainer class instantiation.

### `expr` Folder
The `expr` folder contains the source code for performing experiments. This includes:
- swapping & interpolation
- OOD classification data preparation