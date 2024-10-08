from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic
from typing_extensions import TypeVar
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor, nn
from beyond_backprop.datamodules.datamodule import DataModule

from beyond_backprop.utils.types import PhaseStr, StepOutputDict

if typing.TYPE_CHECKING:
    pass

# Type variable that describes the supported base networks for an algorithm class.
# This is just a fancy way of showing at 'compile-time' that the base network must be one of the
# supported types, otherwise an error will be raised when creating the algorithm.
# NOTE: This is marked as contravariant, because an Algorithm[ResNet] can be used when we need an
# Algorithm[ResNet18], but we can't use an Algorithm[ResNet18] when we need an Algorithm[ResNet].
NetworkType = TypeVar("NetworkType", bound=nn.Module, contravariant=True)

BatchType = TypeVar("BatchType")


class Algorithm(LightningModule, ABC, Generic[NetworkType, BatchType]):
    """Base class for a learning algorithm.

    This is an extension of the LightningModule class from PyTorch Lightning, with some common
    boilerplate code to keep the algorithm implementations as simple as possible.

    The networks themselves are created separately.
    """

    @dataclass
    class HParams:
        """Hyper-parameters of the algorithm."""

    def __init__(
        self,
        datamodule: DataModule[BatchType],
        network: NetworkType,
        hp: Algorithm.HParams | None = None,
    ):
        super().__init__()
        self.datamodule: DataModule[BatchType] = datamodule
        self.hp = hp or self.HParams()
        assert isinstance(network, nn.Module)
        self.network: NetworkType = network

        self.trainer: Trainer

    def training_step(self, batch: BatchType) -> StepOutputDict:
        """Performs a training step."""
        return self.shared_step(batch=batch, phase="train")

    def validation_step(self, batch: BatchType) -> StepOutputDict:
        """Performs a validation step."""
        return self.shared_step(batch=batch, phase="val")

    def test_step(self, batch: BatchType) -> StepOutputDict:
        """Performs a test step."""
        return self.shared_step(batch=batch, phase="test")

    def shared_step(self, batch: BatchType, phase: PhaseStr) -> StepOutputDict:
        """Performs a training/validation/test step.

        This must return a dictionary with at least the 'y' and 'logits' keys, and an optional
        `loss` entry. This is so that the training of the model is easier to parallelize the
        training across GPUs:
        - the cross entropy loss gets calculated using the global batch size
        - the main metrics are logged inside `training_step_end` (supposed to be better for DP/DDP)
        """
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        """Creates the optimizers and the learning rate schedulers."""

    def forward(self, x: Tensor) -> Tensor:
        """Performs a forward pass.

        Feel free to overwrite this to do whatever you'd like.
        """
        return self.network(x)

    def configure_callbacks(self) -> list[Callback]:
        """Use this to add some callbacks that should always be included with the model."""
        if getattr(self.hp, "use_scheduler", False) and self.trainer and self.trainer.logger:
            from lightning.pytorch.callbacks.lr_monitor import LearningRateMonitor

            return [LearningRateMonitor()]
        return []
