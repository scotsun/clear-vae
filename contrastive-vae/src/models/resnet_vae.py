from typing import Callable, List, Optional
import torch
import torch.nn as nn
from torch import Tensor


"""
Modified from https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""


def conv3x3(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
    output_padding: int = 0,
    mode: str = "encoding",
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    match mode:
        case "encoding":
            return nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                bias=False,
                dilation=dilation,
            )
        case "decoding":
            return nn.ConvTranspose2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                output_padding=output_padding,
                groups=groups,
                bias=False,
                dilation=dilation,
            )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    pass
