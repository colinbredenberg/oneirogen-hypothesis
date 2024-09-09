from __future__ import annotations

from logging import getLogger as get_logger

import torch
from lightning import Callback, Trainer
from torch import Tensor

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from beyond_backprop.algorithms.algorithm import Algorithm
from beyond_backprop.utils.types import StepOutputDict
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
logger = get_logger(__name__)

class SimRecord(defaultdict):
    __getattr__= defaultdict.__getitem__
    __setattr__= defaultdict.__setitem__
    __delattr__= defaultdict.__delitem__

class TestRecord(Callback):
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()
        self.record = SimRecord(lambda: [])

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Algorithm, outputs, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        self.record.reward.append(torch.mean(pl_module.forward_network.reward_node.output.detach()))
        return

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: Algorithm, outputs, batch: tuple[Tensor, Tensor], batch_idx: int, 
        data_loader_idx = 0) -> None:

        self.record.output.append(pl_module.forward_network.output.detach())
        self.record.target.append(pl_module.forward_network.target.detach()) #the actual target activity of l2

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        # TODO: Save plots in the log directory and to wandb if used.
        plt.figure()
        plt.plot(torch.tensor(self.record.reward))
        plt.title('reward')
        plt.show()

        output = torch.hstack(self.record.output)
        target = torch.hstack(self.record.target)
        # predicted_target = torch.vstack(test_sim.record.predicted_target)

        plt.figure()
        #plot data against network outputs
        plt.scatter(target[:], output[:])
        plt.plot([-1, 1], [-1, 1], 'k')
        plt.title('I/O comparison')
        plt.show()
        return