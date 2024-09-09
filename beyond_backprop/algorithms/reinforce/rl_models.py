from __future__ import annotations

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.independent import Independent
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.bernoulli import Bernoulli
import networkx as nx
from beyond_backprop.algorithms.common.graph_utils import sequential_graph, multi_stream_graph
from beyond_backprop.algorithms.common.layered_network import LayeredNetwork
from beyond_backprop.networks.network import Network
import beyond_backprop.algorithms.common.layer as layer
from dataclasses import dataclass, field

class RLModel(LayeredNetwork):
    """
    A generic architecture for learning MNIST with
    a REINFORCE-Backprop hybrid
    """
    @dataclass
    class HParams(Network.HParams):
        differentiable: bool = False
    
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: RLModel.HParams | None = None,
    ):
        #Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        response_probs = layer.ProbabilityLayer(lambda x: dist.categorical.Categorical(probs = x), differentiable = hparams.differentiable)
        
        graph = sequential_graph([response_probs])
        super().__init__(graph = graph)
        #modules have to be assigned after super().__init__ call
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)
        self.response_probs = response_probs
        self.reward_node = layer.ClassificationRLLoss()

    def forward(self,x: tuple[Tensor, Tensor]) -> Tensor:
        self.target = x[1] #pass through targets
        x = x[0] #pass through inputs
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output_probs = F.softmax(x) #probabilities for output
        self.output = self.response_probs.forward(output_probs)
        self.reward_node(self.output,self.target)
        return output_probs

class ProbRLModel(LayeredNetwork):
    """
    A generic architecture for learning MNIST with a
    layer-wise REINFORCE model
    """
    def __init__(self, differentiable = False):
        #Graph nodes have to be assigned before super().__init__ call
        response_probs = layer.ProbabilityLayer(lambda x: dist.categorical.Categorical(probs = x), differentiable = differentiable)
        layer_noise = layer.ProbabilityLayer(lambda x: dist.normal.Normal(x, 0.01))
        graph = sequential_graph([layer_noise, response_probs])
        super().__init__(graph = graph)
        #modules have to be assigned after super().__init__ call
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)
        self.layer_noise = layer_noise
        self.fc1 = nn.Linear(320,50)
        self.fc2 = nn.Linear(50,10)
        self.response_probs = response_probs
        self.loss = layer.ClassificationRLLoss()

    def forward(self,x: tuple[Tensor, Tensor]) -> Tensor:
        self.target = x[1] #pass through targets
        x = x[0] #pass through inputs
        x = F.relu(F.max_pool2d(self.conv1(x),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = x.view(-1, 320)
        x = self.layer_noise(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        output_probs = F.softmax(x) #probabilities for output
        self.output = self.response_probs(output_probs)
        self.loss(self.output,self.target)
        return self.output