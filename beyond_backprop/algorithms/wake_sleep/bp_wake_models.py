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

def generic_tanh_layer(in_dim, dim, out_dim, sigma_inf, sigma_gen):
    funcs = {
        "inf": nn.Sequential(nn.Linear(in_dim, dim), nn.Tanh()),
        "gen": nn.Sequential(nn.Linear(out_dim, dim), nn.Tanh()),
    }
    dists = [
        lambda x: Independent(Normal(x, sigma_inf), 1),
        lambda x: Independent(dist.normal.Normal(x, sigma_gen), 1),
    ]
    l = layer.IGFuncLayerWrapper(funcs, dists)
    return l


def generic_feedback_layer(in_dim, dim, out_dim, sigma_inf, sigma_gen, flatten=False, differentiable = False):
    gen_mean_func = nn.Sequential(nn.Linear(out_dim, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Tanh())
    gen_var_func = nn.Sequential(nn.Linear(out_dim, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, dim), nn.Sigmoid())
    if flatten:
        mean_func = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, dim), nn.Tanh())
    else:
        mean_func = nn.Sequential(nn.Linear(in_dim, dim), nn.Tanh())
    funcs = {
        "inf": mean_func,
        "gen": layer.MeanScale(gen_mean_func, gen_var_func, epsilon=0.001),
        # "gen" : gen_mean_func
    }
    dists = [
        lambda x: Independent(Normal(x, sigma_inf), 1),
        lambda x: Independent(dist.normal.Normal(x[0], x[1]), 1),
        # lambda x: Independent(Normal(x, sigma_gen), 1)
    ]
    l = layer.IGFuncLayerWrapper(funcs, dists, differentiable = differentiable)
    return l

class FCGenNet(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128, 24)
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: FCGenNet.HParams | None = None,
    ):
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.l1_N = int(np.prod(self.hparams.input_shape))
        l1_gen_mean_func = nn.Sequential(
            nn.Linear(hparams.layer_widths[0], self.l1_N),
            nn.Tanh(),
            nn.Unflatten(-1, hparams.input_shape),
        )
        l1_funcs = {
            "inf": layer.MeanPlusNoise(hparams.input_shape, nn.Identity(), hparams.sigma_inf),
            "gen": layer.MeanPlusNoise(hparams.input_shape, l1_gen_mean_func, hparams.sigma_gen),
        }
        l1_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0], differentiable = True)
        layer_list = [l1]

        lN_mean_func = nn.Sequential(nn.Linear(hparams.layer_widths[-2], hparams.layer_widths[-1]))
        # lN_var_func = nn.Sequential(nn.Linear(hparams.layer_widths[-2], hparams.layer_widths[-1]), nn.Sigmoid())
        # lN_funcs = {"inf": layer.MeanScale(lN_mean_func, lN_var_func, epsilon = 0.001), "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size, 0, 1)}
        # lN_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
        lN_funcs = {
            "inf": lN_mean_func,
            "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        }
        lN_dists = [
            lambda x: OneHotCategorical(logits=x),
            lambda x: OneHotCategorical(logits=x[0]),
        ]
        lN = layer.IGFuncLayerWrapper(lN_funcs, lN_dists)

        for layer_idx, width in enumerate(hparams.layer_widths):
            if layer_idx == 0:
                layer_list.append(
                    generic_feedback_layer(
                        self.l1_N,
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                        flatten=True,
                        differentiable = True
                    )
                )
            elif layer_idx == len(hparams.layer_widths) - 1:
                layer_list.append(lN)
            else:
                layer_list.append(
                    generic_feedback_layer(
                        hparams.layer_widths[layer_idx - 1],
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                        differentiable = True
                    )
                )

        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())

class NoiselessFCGenNet(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128, 24)
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: FCGenNet.HParams | None = None,
    ):
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.l1_N = int(np.prod(self.hparams.input_shape))
        l1_gen_mean_func = nn.Sequential(
            nn.Linear(hparams.layer_widths[0], self.l1_N),
            nn.Tanh(),
            nn.Unflatten(-1, hparams.input_shape),
        )
        l1_funcs = {
            "inf": layer.MeanPlusNoise(hparams.input_shape, nn.Identity(), hparams.sigma_inf),
            "gen": layer.MeanPlusNoise(hparams.input_shape, l1_gen_mean_func, hparams.sigma_gen),
        }
        l1_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0], differentiable = True)
        network_list = [nn.Sequential(nn.Linear(hparams.layer_widths[ii-1], hparams.layer_widths[ii]),nn.Tanh()) for ii in range(1, len(hparams.layer_widths)-1)]
        lN_prev_mean_func = nn.Sequential(network_list)
        lN_prev_gen_mean_func = nn.Sequential([nn.Linear(hparams.layer_widths[-1], hparams.layer_widths[-2]), nn.Tanh()])
        lN_prev_gen_var_func = nn.Sequential([nn.Linear(hparams.layer_widths[-1], hparams.layer_widths[-2]), nn.Sigmoid()])

        lN_prev_funcs = {
            "inf": layer.MeanPlusNoise(hparams.input_shape, lN_prev_mean_func, hparams.sigma_inf),
            "gen": layer.MeanScale(lN_prev_gen_mean_func, lN_prev_gen_var_func, epsilon=0.001),
        }
        lN_prev_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1)
        ]
        lN_prev = layer.IGFuncLayerWrapper(lN_prev_funcs, lN_prev_dists)
        lN_mean_func = nn.Sequential(nn.Linear(hparams.layer_widths[-2], hparams.layer_widths[-1]), nn.Tanh())
        # lN_var_func = nn.Sequential(nn.Linear(hparams.layer_widths[-2], hparams.layer_widths[-1]), nn.Sigmoid())
        # lN_funcs = {"inf": layer.MeanScale(lN_mean_func, lN_var_func, epsilon = 0.001), "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size, 0, 1)}
        # lN_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
        lN_funcs = {
            "inf": lN_mean_func,
            "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        }
        lN_dists = [
            lambda x: OneHotCategorical(logits=x),
            lambda x: OneHotCategorical(logits=x[0]),
        ]
        lN = layer.IGFuncLayerWrapper(lN_funcs, lN_dists)
        layer_list = [l1, lN_prev, lN]

        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())