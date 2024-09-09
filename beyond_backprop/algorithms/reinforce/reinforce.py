from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar
import networkx as nx

import torch
import wandb
from hydra.utils import HydraConfig
from hydra_zen import instantiate
from torch import Tensor, nn
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from lightning import Callback, LightningModule, Trainer

from beyond_backprop.configs.lr_scheduler import CosineAnnealingLRConfig
from beyond_backprop.configs.optimizer import SGDConfig
from beyond_backprop.configs.optimizer import AdamConfig
from beyond_backprop.datamodules.image_classification import (
    ImageClassificationDataModule,
)
from beyond_backprop.networks.conv_architecture import (
    get_all_forward_activations_and_indices,
)
from beyond_backprop.networks.invertible import set_input_output_shapes_on_forward
from beyond_backprop.networks.layers import Sequential
from beyond_backprop.networks.lenet import LeNet
from beyond_backprop.networks.resnet import ResNet18, ResNet34
from beyond_backprop.networks.simple_vgg import SimpleVGG
from beyond_backprop.utils.hydra_utils import Partial, add_attributes
from beyond_backprop.utils.utils import is_trainable
from beyond_backprop.algorithms.image_classification import ImageClassificationAlgorithm

from ..algorithm import Algorithm, PhaseStr, StepOutputDict
from beyond_backprop.algorithms.common.layered_network import layer
from beyond_backprop.algorithms.common.layered_network import LayeredNetwork

REINFORCENetworkType = TypeVar("REINFORCENetworkType", LeNet, SimpleVGG, ResNet18, ResNet34)

class REINFORCE(ImageClassificationAlgorithm[REINFORCENetworkType]):
    """
    Implementation of the REINFORCE algorithm (Williams 1992). Works for networks with only stochastic outputs (like DQN or AGREL)
    as well as networks where arbitrary nodes are stochastic rather than deterministic. The network graph defines which nodes use REINFORCE,
    and all other differentiable nodes are backpropagated through.
    """
    @dataclass
    class HParams(ImageClassificationAlgorithm.HParams):
        """Hyper-Parameters of the model.
        """
        reward_baseline: int = 0
        optimizer: AdamConfig = AdamConfig(lr=3e-4)

    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: REINFORCENetworkType,
        hp: REINFORCE.HParams,
    ):
        super().__init__(datamodule=datamodule, network=network, hp=hp)
        self.hp: REINFORCE.HParams
        self.reward_baseline = self.hp.reward_baseline

    def update_baseline(self):
        self.reward_baseline = self.hp.reward_baseline

    def make_forward_network(self, base_network: REINFORCENetworkType) -> LayeredNetwork:
        """Creates the forward network by adapting the base network."""
        assert isinstance(base_network, LayeredNetwork)
        self.check_network(base_network) #verify that the network satisfies the basic requirements for the REINFORCE algorithm
        forward_net = base_network #create_forward_network(base_network)
        forward_net.to(self.device)
        return forward_net

    def make_feedback_network(
        self, base_network: REINFORCENetworkType, forward_network: Sequential
    ) -> None:
        """Creates the feedback network based on the base and forward networks."""
        # NOTE: REINFORCE doesn't require a feedback network, so here we output None

        return nn.Sequential() #Return empty feedback network (nn.Sequential w/ nothing in it)

    def configure_optimizers(self):
        forward_optim = self._create_optimizer()
        # NOTE: Here I'm using a trick so that we can do multiple feedback training iterations
        # on a single batch and still use PyTorch-Lightning automatic optimization:
        return [
            forward_optim,
        ]


    def shared_step(
        self,
        batch: tuple[Tensor, Tensor],
        batch_idx: int,
        phase: PhaseStr,
    ) -> StepOutputDict:
        """Calculates the loss for the current optimizer for the given batch.

        Returns a dictionary with some items that are used by the base class.
        """
        x, y = batch
        training = phase == "train"

        logits = self.network(batch)

        batch_loss = 0.0

        reward = self.network.reward_node.output.flatten() #(self.net.output ==  self.net.target).float()
        #reward_baseline = torch.mean(reward)
        total_log_prob = torch.stack(list(node.log_prob() for node in self.network.graph.nodes)).sum(dim = 0)
        self.pre_grad = torch.mean((reward - self.reward_baseline) * - total_log_prob)

        self.network.loss = -torch.mean(reward)

        batch_loss = self.pre_grad

        return StepOutputDict(logits=logits.detach(), y=y, loss=batch_loss, log={})

    def _create_optimizer(self) -> torch.optim.Optimizer:
        forward_optim_fn = instantiate(self.hp.optimizer)

        forward_optim = forward_optim_fn(self.network.parameters())
        return forward_optim

    def configure_callbacks(self) -> list[Callback]:
        # NOTE: Can actually reuse this:
        from beyond_backprop.algorithms.reinforce.callbacks import TestRecord
        # from beyond_backprop.algorithms.example_target_prop.callbacks import DetectIfTrainingCollapsed
        return super().configure_callbacks() + [
            # DetectIfTrainingCollapsed(),
        ]


    def check_network(self, net):
        assert isinstance(net.graph, nx.DiGraph) #verify that the graph fed in for initialization is a valid directed graph
        assert nx.is_directed_acyclic_graph(net.graph) #verify that the graph is a valid directed acyclic graph
        #verify that each node in the graph is a valid instance of the ProbabilityLayer class
        for node in net.graph.nodes:
            assert isinstance(node, layer.ProbabilityLayer) or isinstance(node, layer.InfGenProbabilityLayer)
            #check to validate that the nodes aren't using rsample (that could cause problems for the differentiation.)
            assert not(node.differentiable) 
            
        #check that every nondifferentiable ProbabilityLayer is included in the graph nodes
        for module in net.modules():
            if not(module in net.graph.nodes):
                assert not(isinstance(module, layer.ProbabilityLayer) and module.differentiable == False)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)