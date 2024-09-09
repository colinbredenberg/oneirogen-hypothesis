from __future__ import annotations
from beyond_backprop.algorithms.algorithm import Algorithm, BatchType, NetworkType
from torch import nn

from beyond_backprop.datamodules.datamodule import DataModule


class AlgorithmWithFeedbackNetwork(Algorithm[NetworkType, BatchType]):
    def __init__(
        self, datamodule: DataModule, network: NetworkType, hp: Algorithm.HParams | None = None
    ):
        super().__init__(datamodule=datamodule, network=network, hp=hp)
        # TODO: Remove these two attributes, maybe move them to a new Algorithm base class.
        self.forward_network: nn.Module  # = self.make_forward_network(base_network=network)
        # NOTE: not sure if this belongs in the algorithm base class. Perhaps some algorithms don't
        # have a feedback network?
        self.feedback_network: nn.Module  # = self.make_feedback_network(
        #     base_network=network, forward_network=self.forward_network
        # )

    def make_forward_network(self, base_network: NetworkType) -> nn.Module:
        """Creates the forward network by optionally adapting the base network."""
        return base_network

    def make_feedback_network(
        self, base_network: nn.Module, forward_network: nn.Module
    ) -> nn.Module:
        """Creates the feedback network based on the base and forward networks."""
        return nn.Sequential()
