from __future__ import annotations

from logging import getLogger as get_logger

import torch
from lightning import Callback, Trainer
from torch import Tensor

from beyond_backprop.algorithms.algorithm import Algorithm
from beyond_backprop.utils.types import StepOutputDict

logger = get_logger(__name__)


class DetectIfTrainingCollapsed(Callback):
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: Algorithm,
        outputs: list[StepOutputDict],
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
    ) -> None:
        x, y = batch

        logits = torch.concat([output["logits"] for output in outputs])
        y_pred = torch.argmax(logits, -1)
        unique_outputs, output_counts = torch.unique(y_pred, return_counts=True)
        unique_y, y_counts = torch.unique(y, return_counts=True)

        # Training has collapsed if the forward network predicts a single class for all samples
        # NOTE: Need to be a little bit careful, what if the batch only contains that class?
        only_predicting_one_class = output_counts[0] / y_pred.shape[0]
        most_common_class_ratio = y_counts[0] / y.shape[0]
        chance_level = 1.0 / pl_module.datamodule.num_classes

        pl_module.log(
            "most_predicted_output_batch_fraction",
            output_counts[0] / y_pred.shape[0],
            prog_bar=True,
        )

        if only_predicting_one_class and most_common_class_ratio <= 1.5 * chance_level:
            # If we're predicting only one class, but the batch contains roughly chance-level
            # distribution of classes, then training has collapsed.
            logger.error(
                f"Training seems to have collapsed: Model is only predicting class "
                f"{unique_outputs[0]}, but the most common class in the batch is class "
                f"{unique_y[0]} that only accounts for {most_common_class_ratio:.2%} of the "
                f"batch."
            )
            trainer.should_stop = True
            raise KeyboardInterrupt(
                f"Training seems to have collapsed: Model is only predicting class "
                f"{unique_outputs[0]}, but the most common class in the batch is class "
                f"{y_counts[0]} that only accounts for {most_common_class_ratio:.2%} of the "
                f"batch."
            )
