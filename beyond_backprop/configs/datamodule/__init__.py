import os
from dataclasses import dataclass, field
from logging import getLogger as get_logger
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from hydra.core.config_store import ConfigStore
from hydra_zen import hydrated_dataclass, instantiate
from torch import Tensor
from torchvision import transforms

from beyond_backprop.datamodules import (
    CIFAR10DataModule,
    FashionMNISTDataModule,
    ImageNet32DataModule,
    MNISTDataModule,
    VisionDataModule,
    cifar10_normalization,
    imagenet32_normalization,
)
from beyond_backprop.datamodules.inaturalist import INaturalistDataModule, TargetType, Version

logger = get_logger(__name__)


FILE = Path(__file__)


REPO_ROOTDIR = FILE.parent.parent.parent  # The root of the repo.
for level in range(5):
    if "README.md" in list(p.name for p in REPO_ROOTDIR.iterdir()):
        break
    REPO_ROOTDIR = REPO_ROOTDIR.parent


SLURM_TMPDIR: Optional[Path] = (
    Path(os.environ["SLURM_TMPDIR"]) if "SLURM_TMPDIR" in os.environ else None
)
SLURM_JOB_ID: Optional[int] = (
    int(os.environ["SLURM_JOB_ID"]) if "SLURM_JOB_ID" in os.environ else None
)

TORCHVISION_DIR: Optional[Path] = None

_torchvision_dir = Path("/network/datasets/torchvision")
if _torchvision_dir.exists() and _torchvision_dir.is_dir():
    TORCHVISION_DIR = _torchvision_dir

if not SLURM_TMPDIR and SLURM_JOB_ID is not None:
    # This can happens when running the integrated VSCode terminal with `mila code`!
    _slurm_tmpdir = Path(f"/Tmp/slurm.{SLURM_JOB_ID}.0")
    if _slurm_tmpdir.exists():
        SLURM_TMPDIR = _slurm_tmpdir
DATA_DIR = Path(os.environ.get("DATA_DIR", SLURM_TMPDIR or (REPO_ROOTDIR / "data")))

NUM_WORKERS = int(
    os.environ.get(
        "SLURM_CPUS_PER_TASK",
        os.environ.get(
            "SLURM_CPUS_ON_NODE",
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else torch.multiprocessing.cpu_count(),
        ),
    )
)
logger = get_logger(__name__)


Transform = Callable[[Tensor], Tensor]


@dataclass
class DataModuleConfig:
    ...


@hydrated_dataclass(target=VisionDataModule, populate_full_signature=True)
class VisionDataModuleConfig(DataModuleConfig):
    data_dir: Optional[str] = str(TORCHVISION_DIR or DATA_DIR)
    val_split: Union[int, float] = 0.1  # NOTE: reduced from default of 0.2
    num_workers: int = NUM_WORKERS
    normalize: bool = True  # NOTE: Set to True by default instead of False
    batch_size: int = 32
    seed: int = 42
    shuffle: bool = True  # NOTE: Set to True by default instead of False.
    pin_memory: bool = True  # NOTE: Set to True by default instead of False.
    drop_last: bool = False

    __call__ = instantiate


def mnist_normalization():
    return transforms.Normalize(mean=0.5, std=0.5)


def mnist_train_transforms():
    return transforms.Compose(
        [
            transforms.RandomCrop(size=28, padding=4, padding_mode="edge"),
            transforms.ToTensor(),
            mnist_normalization(),
        ]
    )


@hydrated_dataclass(target=mnist_train_transforms)
class MNISTTrainTransforms:
    ...


@hydrated_dataclass(target=MNISTDataModule, populate_full_signature=True)
class MNISTDataModuleConfig(VisionDataModuleConfig):
    normalize: bool = True
    batch_size: int = 128
    train_transforms: MNISTTrainTransforms = field(default_factory=MNISTTrainTransforms)


def cifar10_train_transforms():
    return transforms.Compose(
        [
            transforms.RandomCrop(size=32, padding=4, padding_mode="edge"),
            transforms.ToTensor(),
            cifar10_normalization(),
        ]
    )


@hydrated_dataclass(target=cifar10_train_transforms)
class Cifar10TrainTransforms:
    ...


@hydrated_dataclass(target=CIFAR10DataModule, populate_full_signature=True)
class CIFAR10DataModuleConfig(VisionDataModuleConfig):
    train_transforms: Cifar10TrainTransforms = field(default_factory=Cifar10TrainTransforms)
    # Overwriting this one:
    batch_size: int = 128

cs = ConfigStore.instance()
cs.store(group="datamodule", name="base", node=DataModuleConfig)
cs.store(group="datamodule", name="cifar10", node=CIFAR10DataModuleConfig)
cs.store(group="datamodule", name="mnist", node=MNISTDataModuleConfig)