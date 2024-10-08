from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy
from beyond_backprop.algorithms.algorithm import Algorithm, NetworkType

from beyond_backprop.datamodules.image_classification import (
    ImageClassificationDataModule,
)
from beyond_backprop.utils.types import ClassificationOutputs, PhaseStr, StepOutputDict
from beyond_backprop.utils.utils import get_device


class ImageClassificationAlgorithm(Algorithm[NetworkType, tuple[Tensor, Tensor]]):
    """Base class for a learning algorithm.

    This is an extension of the LightningModule class from PyTorch Lightning, with some common
    boilerplate code to keep the algorithm implementations as simple as possible.

    The networks themselves are created separately.

    ## TODOs:
    - Once a non-image-classification algorithm comes up, move all the image-classification-related
    things to a new intermediate subclass, such as `ImageClassificationAlgorithm`.
    """

    @dataclass
    class HParams(Algorithm.HParams):
        """Hyper-parameters of the algorithm."""

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: NetworkType,
        hp: ImageClassificationAlgorithm.HParams | None = None,
    ):
        super().__init__(datamodule=datamodule, network=network, hp=hp)
        self.datamodule: ImageClassificationDataModule
        # NOTE: Setting this property allows PL to infer the shapes and number of params.
        device = get_device(network)
        self.example_input_array = torch.rand(
            [datamodule.batch_size, *datamodule.dims],
            device=device,
        )
        num_classes: int = datamodule.num_classes

        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.train_top5_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.val_top5_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5)
        self.test_top5_accuracy = MulticlassAccuracy(num_classes=num_classes, top_k=5)

    def predict(self, x: Tensor) -> Tensor:
        """Predict the classification labels."""
        return self.network(x).argmax(-1)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> ClassificationOutputs:
        """Performs a training step."""
        return self.shared_step(batch=batch, batch_idx=batch_idx, phase="train")

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> ClassificationOutputs:
        """Performs a validation step."""
        return self.shared_step(batch=batch, batch_idx=batch_idx, phase="val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> ClassificationOutputs:
        """Performs a test step."""
        return self.shared_step(batch=batch, batch_idx=batch_idx, phase="test")

    @abstractmethod
    def shared_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int, phase: PhaseStr
    ) -> ClassificationOutputs:
        """Performs a training/validation/test step.

        This must return a dictionary with at least the 'y' and 'logits' keys, and an optional
        `loss` entry. This is so that the training of the model is easier to parallelize the
        training across GPUs:
        - the cross entropy loss gets calculated using the global batch size
        - the main metrics are logged inside `training_step_end` (supposed to be better for DP/DDP)
        """

    def training_step_end(self, step_output: StepOutputDict) -> StepOutputDict:
        """Called with the results of each worker / replica's output.

        See the `training_step_end` of pytorch-lightning for more info.
        """
        return self.shared_step_end(step_output, phase="train")

    def validation_step_end(self, step_output: StepOutputDict) -> StepOutputDict:
        return self.shared_step_end(step_output, phase="val")

    def test_step_end(self, step_output: StepOutputDict) -> StepOutputDict:
        return self.shared_step_end(step_output, phase="test")

    def shared_step_end(self, step_output: StepOutputDict, phase: PhaseStr) -> StepOutputDict:
        required_entries = ("logits", "y")
        if not isinstance(step_output, dict):
            raise RuntimeError(
                f"Expected the {phase} step method to output a dictionary with at least the "
                f"{required_entries} keys, but got an output of type {type(step_output)} instead!"
            )
        if not all(k in step_output for k in required_entries):
            raise RuntimeError(
                f"Expected all the following keys to be in the output of the {phase} step "
                f"method: {required_entries}"
            )
        logits = step_output["logits"]
        y = step_output["y"]

        probs = torch.softmax(logits, -1)

        accuracy = getattr(self, f"{phase}_accuracy")
        top5_accuracy = getattr(self, f"{phase}_top5_accuracy")

        assert isinstance(accuracy, MulticlassAccuracy)
        assert isinstance(top5_accuracy, MulticlassAccuracy)

        accuracy(probs, y)
        top5_accuracy(probs, y)

        prog_bar = phase == "train"
        self.log(f"{phase}/accuracy", accuracy, prog_bar=prog_bar, sync_dist=True)
        self.log(f"{phase}/top5_accuracy", top5_accuracy, prog_bar=prog_bar, sync_dist=True)
        
        if "cross_entropy" not in step_output:
            # Add the cross entropy loss as a metric.
            ce_loss = F.cross_entropy(logits.detach(), y, reduction="mean")
            self.log(f"{phase}/cross_entropy", ce_loss, prog_bar=prog_bar, sync_dist=True)

        fused_output = step_output.copy()
        loss: Tensor | float | None = step_output.get("loss", None)
        self.log(f"{phase}/loss", loss, prog_bar = prog_bar, sync_dist = True)
        if isinstance(loss, Tensor) and loss.shape:
            # Replace the loss with its mean. This is useful when automatic
            # optimization is enabled, for example in the baseline (backprop), where each replica
            # returns the un-reduced cross-entropy loss. Here we need to reduce it to a scalar.
            fused_output["loss"] = loss.mean()

        return fused_output
