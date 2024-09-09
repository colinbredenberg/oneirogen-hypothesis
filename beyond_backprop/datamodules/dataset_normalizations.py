from typing import Callable

from torchvision import transforms
from torch import Tensor
import torch

def imagenet_normalization() -> Callable:
    return transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def cifar10_normalization() -> Callable:
    return transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
    )

def cifar10_unnormalization(x) -> Tensor:
    mean= torch.Tensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).view([1,1,3])
    std= torch.Tensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).view([1,1,3])
    return (x * std) + mean
    # return x

def mnist_unnormalization(x) -> Tensor:
    mean = 0.5
    std = 0.5
    return (x * std) + mean

def stl10_normalization() -> Callable:
    return transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27))


def emnist_normalization(split: str) -> Callable:
    # `stats` contains mean and std for each `split`.
    stats = {
        "balanced": (0.175, 0.333),
        "byclass": (0.174, 0.332),
        "bymerge": (0.174, 0.332),
        "digits": (0.173, 0.332),
        "letters": (0.172, 0.331),
        "mnist": (0.173, 0.332),
    }
    return transforms.Normalize(mean=stats[split][0], std=stats[split][1])
