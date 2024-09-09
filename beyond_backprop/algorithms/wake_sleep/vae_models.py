from __future__ import annotations

from torch import nn
import numpy as np
import torch.distributions as dist
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
from torch.distributions.one_hot_categorical import OneHotCategorical
import networkx as nx
import beyond_backprop.algorithms.common.layer as layer
from beyond_backprop.algorithms.common.graph_utils import sequential_graph
from .inf_gen_network import InfGenNetwork
from dataclasses import dataclass
from beyond_backprop.networks.network import Network

class VAEFCGenNet(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128, 24)
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)
        differentiable: bool = True

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: VAEFCGenNet.HParams | None = None,
    ):
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.l1_N = int(np.prod(self.hparams.input_shape))
        layer_widths = [self.l1_N, *hparams.layer_widths]
        gen_network_list = [nn.Sequential(nn.Linear(layer_widths[ii], layer_widths[ii-1]),nn.Tanh()) for ii in range(len(layer_widths)-1, 0, -1)]
        l1_gen_mean_func = nn.Sequential(*gen_network_list, nn.Unflatten(-1, hparams.input_shape))
        l1_funcs = {
            "inf": layer.MeanPlusNoise(hparams.input_shape, nn.Identity(), hparams.sigma_inf),
            "gen": layer.MeanPlusNoise(hparams.input_shape, l1_gen_mean_func, hparams.sigma_gen),
        }
        l1_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1),
        ]
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0], differentiable = hparams.differentiable)
        network_list = [nn.Sequential(nn.Linear(layer_widths[ii-1], layer_widths[ii]),nn.Tanh()) for ii in range(1, len(layer_widths)-1)]
        lN_mean_func = nn.Sequential(nn.Flatten(), *network_list, nn.Linear(layer_widths[-2],layer_widths[-1]), nn.Tanh())
        lN_scale_func = nn.Sequential(nn.Flatten(), *network_list, nn.Linear(layer_widths[-2], layer_widths[-1]))
        #var_network_list = [nn.Sequential(nn.Linear(layer_widths[ii-1], layer_widths[ii]),nn.Tanh()) for ii in range(1, len(layer_widths)-1)]
        #var_network_list.append(nn.Sequential(nn.Linear(layer_widths[-2], layer_widths[-1]), nn.Sigmoid()))
        #lN_var_func = nn.Sequential(nn.Flatten(), *var_network_list)
        lN_funcs = {
            "inf": layer.MeanExpScale(lN_mean_func, lN_scale_func),
            "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        }
        lN_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1)
        ]
        lN = layer.StandardNormalIGFuncLayerWrapper(lN_funcs, lN_dists, differentiable = hparams.differentiable)

        layer_list = [l1, lN]

        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())

class LadderVAEFCGenNet(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128, 24)
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)
        differentiable: bool = True

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: VAEFCGenNet.HParams | None = None,
    ):
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.l1_N = int(np.prod(self.hparams.input_shape))
        layer_widths = [self.l1_N, *hparams.layer_widths]
        l1_gen_mean_func = nn.Sequential(nn.Linear(layer_widths[1], layer_widths[0]), nn.Tanh(), nn.Unflatten(-1, hparams.input_shape))
        l1_funcs = {
            "inf": layer.MeanPlusNoise(hparams.input_shape, nn.Identity(), hparams.sigma_inf),
            "gen": layer.MeanPlusNoise(hparams.input_shape, l1_gen_mean_func, hparams.sigma_gen),
        }
        l1_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1),
        ]
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0], differentiable = hparams.differentiable)

        gen_network_list = [nn.Sequential(nn.Linear(layer_widths[ii+1], layer_widths[ii]),nn.Tanh()) for ii in range(1, len(layer_widths)-1)]
        network_list = [nn.Sequential(nn.Linear(layer_widths[ii-1], layer_widths[ii]),nn.Tanh()) for ii in range(1, len(layer_widths)-1)]
        layer_list = []
        for l in range(0, len(network_list)):
            if l == 0:
                mean_func = nn.Sequential(nn.Flatten(), network_list[l])
            else:
                mean_func = network_list[l]
            gen_mean_func = gen_network_list[l]
            layer_funcs = {
                "inf": layer.MeanPlusNoise(layer_widths[l+1], mean_func, hparams.sigma_inf),
                "gen": layer.MeanPlusNoise(layer_widths[l+1], gen_mean_func, hparams.sigma_gen)
            }
            layer_dists = [
                lambda x: Independent(Normal(x[0], x[1], 1),1),
                lambda x: Independent(Normal(x[0], x[1], 1),1),
            ]
            layer_list.append(layer.IGFuncLayerWrapper(layer_funcs, layer_dists, differentiable = hparams.differentiable))
        
        lN_mean_func = nn.Sequential(nn.Linear(layer_widths[-2],layer_widths[-1]), nn.Tanh())
        lN_scale_func = nn.Sequential(nn.Linear(layer_widths[-2], layer_widths[-1]))
        #var_network_list = [nn.Sequential(nn.Linear(layer_widths[ii-1], layer_widths[ii]),nn.Tanh()) for ii in range(1, len(layer_widths)-1)]
        #var_network_list.append(nn.Sequential(nn.Linear(layer_widths[-2], layer_widths[-1]), nn.Sigmoid()))
        #lN_var_func = nn.Sequential(nn.Flatten(), *var_network_list)
        lN_funcs = {
            "inf": layer.MeanExpScale(lN_mean_func, lN_scale_func),
            "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        }
        lN_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1)
        ]
        lN = layer.StandardNormalIGFuncLayerWrapper(lN_funcs, lN_dists, differentiable = hparams.differentiable)

        layer_list = [l1, *layer_list, lN]

        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())