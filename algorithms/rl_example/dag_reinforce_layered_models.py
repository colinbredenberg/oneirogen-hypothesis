from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.independent import Independent
from torch.distributions.laplace import Laplace
from torch.distributions.one_hot_categorical import OneHotCategorical
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.gamma import Gamma
from torch.distributions.half_normal import HalfNormal
import networkx as nx
import beyond_backprop.algorithms.common.layer as layer
from beyond_backprop.algorithms.common.graph_utils import sequential_graph, multi_stream_graph
from dataclasses import dataclass, field
from beyond_backprop.networks.network import Network
from beyond_backprop.algorithms.common.layered_network import LayeredNetwork

class DAGRLModel(LayeredNetwork):
    """Network composed of 3 sequential IGFuncLayerWrapper layers for performing the Wake-Sleep algorithm on MNIST"""
    @dataclass
    class HParams(Network.HParams):
        l1_N: int = 0
        l2_N: int = 0

    def __init__(
        self,
        input_shape: int,
        output_shape:int,
        in_channels: int | None = None,
        n_classes: int | None = None,
        hparams: DAGRLModel.HParams | None = None,
    ):
        #Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.input_shape = (input_shape,)
        self.input_dims = input_shape
        self.output_shape = (output_shape,)
        self.output_dims = output_shape
        #declare the functions and distributions operating at each layer of the network
        
        l_out_func = nn.Sequential(layer.View([1, self.input_dims], no_batch = True),
            layer.FALinear(self.input_dims, self.hparams.l1_N),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            layer.FALinear(self.hparams.l1_N, self.hparams.l2_N),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            layer.FALinear(self.hparams.l2_N, self.output_dims)
        )
        l_out_dist = lambda x: dist.categorical.Categorical(logits = x)
        l_out = layer.ProbModuleLayer(l_out_dist, l_out_func)

        #construct the generative and inference graphs
        graph = sequential_graph([l_out])

        #feed the graphs into the superclass initialization
        super().__init__(graph=graph)

        #assign each layer as a submodule of the network (it will be seen by the nn.Module functions)
        self.l_out = l_out