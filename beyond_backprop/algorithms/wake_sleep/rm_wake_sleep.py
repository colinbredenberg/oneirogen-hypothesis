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
import beyond_backprop.algorithms.common.layer as layer
from beyond_backprop.algorithms.common.layered_network import LayeredNetwork
from .inf_gen_network import InfGenNetwork

import networkx as nx
import torch
import torch.nn.functional as F

WakeSleepNetworkType = TypeVar("WakeSleepNetworkType", InfGenNetwork,LeNet, SimpleVGG, ResNet18, ResNet34)

class RMWakeSleep(ImageClassificationAlgorithm[WakeSleepNetworkType]):
    """Implementation of the Wake-Sleep algorithm. Works for networks w/ a VAE-like architecture (only the last layer is latent)
    as well as networks where arbitrary nodes are stochastic rather than deterministic. Requires TWO graphs: one for the inference pass
    and one for the generative pass."""
    @dataclass
    class HParams(ImageClassificationAlgorithm.HParams):
        """Hyper-Parameters of the model.
        """
        forward_optimizer: AdamConfig = AdamConfig(lr=3e-4)
        backward_optimizer: AdamConfig = AdamConfig(lr = 3e-4)
        class_optimizer: AdamConfig = AdamConfig(lr = 3e-4)
        reward_temp: float = 0.01
        burn_in_time: int = 0
        wake_phase_length: int = 1
        sleep_phase_length: int = 1
        sleep_phase_number: int = 0
        wake_loss_ratio: float = 1.
        hallucination_mode: str = 'interp'
    def __init__(
        self,
        datamodule: ImageClassificationDataModule,
        network: WakeSleepNetworkType,
        hp: RMWakeSleep.HParams,
    ):
        super().__init__(datamodule=datamodule, network=network, hp=hp)
        self.hp: RMWakeSleep.HParams
        self.non_input_layers = []
        self.reward_temp = self.hp.reward_temp
        self.burn_in_counter = 0
        self.burn_in_time = self.hp.burn_in_time
        self.wake_phase_counter = 0
        self.sleep_phase_counter = 0
        self.wake_loss_prev = torch.inf
        self.pre_grad_inf = torch.tensor(0.)
        self.pre_grad_gen = torch.tensor(0.)
        self.wake_phase_length = self.hp.wake_phase_length
        self.sleep_phase_length = self.hp.sleep_phase_length
        self.sleep_phase_number = self.hp.sleep_phase_number
        self.wake_loss_ratio = self.hp.wake_loss_ratio
        self.automatic_optimization = False
        self.hallucination_mode = self.hp.hallucination_mode
        for layer in self.network.graph.nodes():
            if layer.input_layer == False:
                self.non_input_layers.append(layer)

        # self.non_input_layers = []
        # for layer in self.network.graph.nodes():
        #     if layer.input_layer == False:
        #         self.non_input_layers.append(layer)

    def make_forward_network(self, base_network: WakeSleepNetworkType) -> InfGenNetwork:
        """Creates the forward network by adapting the base network."""
        assert isinstance(base_network, InfGenNetwork)
        self.check_network(base_network) #verify that the network satisfies the basic requirements for the Wake-Sleep algorithm
        forward_net = base_network #create_forward_network(base_network)
        forward_net.to(self.device)
        return forward_net

    def make_feedback_network(
        self, base_network: WakeSleepNetworkType, forward_network: Sequential
    ) -> None:
        """Creates the feedback network based on the base and forward networks."""
        # NOTE: WakeSleep doesn't require a feedback network, so here we output None

        return nn.Sequential() #Return empty feedback network (nn.Sequential w/ nothing in it)

    def configure_optimizers(self):
        forward_optim, backward_optim, class_optim = self._create_optimizer()
        # NOTE: Here I'm using a trick so that we can do multiple feedback training iterations
        # on a single batch and still use PyTorch-Lightning automatic optimization:
        return [
            forward_optim, backward_optim, class_optim
        ]

    #     self.forward_network.loss = self.pre_grad_gen + self.pre_grad_inf
    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> StepOutputDict:
        return self.shared_step(batch, batch_idx, phase="train")
    
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
        # if self.burn_in_counter == 0:
        self.x = x    
        self.y = y
        training = phase == "train"
        gen_opt, inf_opt, class_opt = self.optimizers()
        #optimize generative weights

        self.network(self.x)

        
        logits = self.network.ts[-1].output
        # reward = (torch.argmax(logits, axis = 1) == y).float().detach()#torch.exp(((torch.argmax(logits, axis = 1) == y).float().detach() - 1)/self.reward_temp)
        # reward = reward#/torch.sum(reward)
        total_likelihood_gen_rm = - self.network.gen_log_prob()#self.network.diagonal_normal_KL(x, gen = True)#- reward * self.network.gen_log_prob() #

        self.pre_grad_gen = torch.mean(total_likelihood_gen_rm)
        self.wake_phase_counter += 1
        #batch_loss = self.pre_grad_gen
        # network_output = self.network.ts[3].output #self.network.ts[4].output
        network_output = self.network.ts[1].output #self.network.ts[4].output

        logits = self.network.classifier(network_output)#F.one_hot(y, num_classes = self.network.n_classes).float()
        #batch_loss = self.pre_grad_gen# + self.pre_grad_inf
        batch_loss = F.nll_loss(logits, y)    

        if training:
            gen_opt.zero_grad()
            self.manual_backward(self.pre_grad_gen)
            gen_opt.step()
            # self.log("reward", torch.mean(reward), prog_bar = True)
            self.log("gen_loss", self.pre_grad_gen, prog_bar = True)

            class_opt.zero_grad()
            self.manual_backward(batch_loss)
            class_opt.step()

            #Calculate parameter updates for the inference parameters
            if self.wake_phase_counter % self.wake_phase_length == 0 and self.burn_in_counter >= self.burn_in_time and self.pre_grad_gen <= self.wake_loss_ratio * self.wake_loss_prev:
                self.wake_loss_prev = self.pre_grad_gen.detach()
                self.sleep_phase_counter += 1
                for ii in range(0,self.sleep_phase_length):
                    self.network.gen_forward()
                    #logits = self.network.l3.gen_output
                    total_likelihood_inf = -self.network.log_prob() #self.network.diagonal_normal_KL()#
                    self.pre_grad_inf = torch.mean(total_likelihood_inf)
                    self.log("inf_loss", self.pre_grad_inf, prog_bar = True)
                    if self.burn_in_counter >= self.burn_in_time:
                        inf_opt.zero_grad()
                        self.manual_backward(self.pre_grad_inf)
                        inf_opt.step()
        self.burn_in_counter += 1
        # if not(training):
        #     batch_loss = 0.
            
        return StepOutputDict(logits=logits.detach(), y=y, loss=batch_loss, log={})

    def _create_optimizer(self) -> torch.optim.Optimizer:
        forward_optim_fn = instantiate(self.hp.forward_optimizer)
        backward_optim_fn = instantiate(self.hp.backward_optimizer)
        class_optim_fn = instantiate(self.hp.class_optimizer)

        forward_optim = forward_optim_fn(self.network.inf_group.parameters())
        backward_optim = backward_optim_fn(self.network.gen_group.parameters())
        class_optim = class_optim_fn(self.network.classifier.parameters())
        return backward_optim, forward_optim, class_optim

    def configure_callbacks(self) -> list[Callback]:
        return super().configure_callbacks() + [
        ]

    def check_network(self, net: nn.Module):
        #CHECK that the network is an instance of a particular class
        #super().check_network(net)
        #assert that both the inference graph and generative graph for the network are directed acyclic graphs
        assert isinstance(net.graph, nx.DiGraph) and isinstance(net.gen_graph, nx.DiGraph)
        assert nx.is_directed_acyclic_graph(net.graph) and nx.is_directed_acyclic_graph(net.gen_graph)

        #assert that both graphs have the same nodes
        assert set(net.graph.nodes) == set(net.gen_graph.nodes)

        for node in net.graph.nodes:
            assert isinstance(node, layer.InfGenProbabilityLayer)
            #check to validate that the nodes aren't using rsample (that could cause problems for the differentiation.)
            assert not(node.differentiable) 
            
        #check that every nondifferentiable ProbabilityLayer is included in the graph nodes
        for module in net.modules():
            if not(module in net.graph.nodes):
                assert not(isinstance(module, layer.ProbabilityLayer) and module.differentiable == False)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)