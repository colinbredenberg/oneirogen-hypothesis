from __future__ import annotations

from torch import nn
import numpy as np
import torch.distributions as dist
import torchvision.transforms.v2 as v2
from torch.distributions.normal import Normal
from torch.distributions.independent import Independent
import networkx as nx
import beyond_backprop.algorithms.common.layer as layer
from beyond_backprop.algorithms.common.graph_utils import sequential_graph
from .inf_gen_network import InfGenNetwork
from dataclasses import dataclass
from beyond_backprop.networks.network import Network


def generic_meannoise_layer(in_dim, dim, out_dim, sigma_inf, sigma_gen, flatten=False, batch_norm = True, dendrites = True):
    """Architecture used for intermediate layers of our single and multicompartment neuron models"""
    branch_num = 16
    gen_dendrite_nl = nn.Tanh()
    inf_dendrite_nl = nn.Tanh()
    if dendrites:
        gen_mean_func_list = [layer.BranchedDendrite(out_dim, branch_num, dim, gen_dendrite_nl, batch_norm = batch_norm)]#
    else:
        gen_mean_func_list = [nn.Linear(out_dim, dim), nn.Tanh()]
        if batch_norm:
            gen_mean_func_list += [nn.BatchNorm1d(dim, affine = False)]

    gen_mean_func = nn.Sequential(*gen_mean_func_list)

    if dendrites:
        mean_func_list = [layer.BranchedDendrite(in_dim, branch_num, dim, inf_dendrite_nl, batch_norm = batch_norm)]
    else:
        mean_func_list = [nn.Linear(in_dim, dim), nn.Tanh()]
        if batch_norm:
             mean_func_list += [nn.BatchNorm1d(dim, affine = False)]

    if flatten:
        mean_func_list = [nn.Flatten(), *mean_func_list]
    mean_func = nn.Sequential(*mean_func_list)
    funcs = {
        "inf": layer.MeanPlusNoise(dim, mean_func, sigma_inf),
        "gen": layer.MeanPlusNoise(dim, gen_mean_func, sigma_gen),
    }

    dists = [
        lambda x: Independent(Normal(x[0], x[1]), 1),
        lambda x: Independent(Normal(x[0], x[1]), 1),
    ]
    l = layer.IGFuncLayerWrapper(funcs, dists)
    return l

def generic_denoise_block(in_dim, dim, out_dim, sigma_inf, sigma_gen, beta, flatten = False):
    """Architecture used for intermediate layers of our recurrent neuron model"""
    diffusion_nl = layer.DiffusionGenNL(dim, 1)
    
    l_denoise_gen_func = layer.DiffusionGen(diffusion_nl, beta)
    mean_func_list = [nn.Linear(in_dim, dim), nn.Tanh()]
    if flatten:
        mean_func_list = [nn.Flatten(), *mean_func_list]
    mean_func = nn.Sequential(*mean_func_list)
    funcs = {
        "inf": layer.MeanPlusNoise(dim, mean_func, sigma_inf),
        "gen": layer.MeanPlusNoise(dim, l_denoise_gen_func, sigma_gen),
    }
    dists = [
        lambda x: Independent(Normal(x[0], x[1]), 1),
        lambda x: Independent(Normal(x[0], x[1]), 1),
    ]
    l = layer.IGFuncLayerWrapper(funcs, dists)
    
    
    gen_mean_func_list = [nn.Linear(out_dim, dim), nn.Tanh()]
    gen_mean_func = nn.Sequential(*gen_mean_func_list)
    l_denoise_mean_func = layer.DiffusionInf(beta)
    l_denoise_funcs = {"inf": layer.MeanPlusNoise(dim, l_denoise_mean_func, np.sqrt(beta)),
                "gen": layer.MeanPlusNoise(dim, gen_mean_func, sigma_gen)}
    l_denoise_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
    l_denoise = layer.IGFuncLayerWrapper(l_denoise_funcs, l_denoise_dists)

    return [l, l_denoise]


class FCWSLayeredModel(InfGenNetwork):
    """InfGenNetwork composed of sequential fully connected layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128, 24)
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)
        n_classes: int = 10
        batch_norm: bool = True
        dendrites: bool = True

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: FCWSLayeredModel.HParams | None = None,
    ):
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.l1_N = int(np.prod(self.hparams.input_shape))
        if self.hparams.input_shape[0] == 1:
            if not(self.hparams.dendrites):
                l1_gen_mean_func = nn.Sequential(
                    nn.Linear(hparams.layer_widths[0], self.l1_N),
                    nn.Tanh(),
                    nn.Unflatten(-1, hparams.input_shape),
                )
            else:
                l1_gen_mean_func = nn.Sequential(
                    layer.BranchedDendrite(hparams.layer_widths[0], 16, self.l1_N, nn.Tanh(), batch_norm = False),
                    nn.Unflatten(-1, hparams.input_shape),
                )
        else:
            l1_gen_mean_func = nn.Sequential(
                nn.Linear(hparams.layer_widths[0], self.l1_N),
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
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0])
        layer_list = [l1]

        if not(self.hparams.batch_norm):
            lN_mean_func = nn.Sequential(nn.Linear(hparams.layer_widths[-2], hparams.layer_widths[-1]))
        else:
            lN_mean_func = nn.Sequential(nn.Linear(hparams.layer_widths[-2], hparams.layer_widths[-1]), nn.BatchNorm1d(hparams.layer_widths[-1], affine = False))
        
        lN_var_func = nn.Sequential(nn.Linear(hparams.layer_widths[-2], hparams.layer_widths[-1]))
        lN_funcs = {"inf": layer.MeanExpScale(lN_mean_func, lN_var_func), "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size, 0, 1)}
        lN_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]

        lN = layer.IGFuncLayerWrapper(lN_funcs, lN_dists)

        for layer_idx, width in enumerate(hparams.layer_widths):
            if layer_idx == 0:
                layer_list.append(
                    generic_meannoise_layer(
                        self.l1_N,
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                        flatten=True,
                        batch_norm = hparams.batch_norm,
                        dendrites = hparams.dendrites,
                    )
                )
            elif layer_idx == len(hparams.layer_widths) - 1:
                layer_list.append(lN)
            else:
                layer_list.append(
                    generic_meannoise_layer(
                        hparams.layer_widths[layer_idx - 1],
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                        batch_norm = hparams.batch_norm,
                        dendrites = hparams.dendrites,
                    )
                )

        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        self.classifier = nn.Sequential(nn.Linear(hparams.layer_widths[0], 256), nn.Tanh(), nn.Linear(256,hparams.n_classes),  nn.LogSoftmax())

class LayerwiseDiffusionModel(InfGenNetwork):
    """InfGenNetwork composed of fully connected layers with within-layer recurrence, of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128, 24)
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        beta: float = 0.1
        N_int: int = 128
        input_shape: tuple = (1, 28, 28)
        n_classes: int = 10

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: LayerwiseDiffusionModel.HParams | None = None,
    ):
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.l1_N = int(np.prod(self.hparams.input_shape))
        if self.hparams.input_shape[0] == 1:
            l1_gen_mean_func = nn.Sequential(
                nn.Linear(hparams.layer_widths[0], self.l1_N),
                nn.Tanh(),
                nn.Unflatten(-1, hparams.input_shape),
            )
        else:
            l1_gen_mean_func = nn.Sequential(
                nn.Linear(hparams.layer_widths[0], self.l1_N),
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
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0])
        layer_list = [l1]

        l_final_mean_func = layer.DiffusionInf(hparams.beta)
        l_final_funcs = {"inf": layer.MeanPlusNoise(hparams.layer_widths[-1], l_final_mean_func, np.sqrt(hparams.beta)), "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size, 0, 1)}
        l_final_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]

        l_final = layer.IGFuncLayerWrapper(l_final_funcs, l_final_dists)

        #construct autoencoder layers
        for layer_idx, width in enumerate(hparams.layer_widths):
            if layer_idx == 0:
                layer_list.extend(
                    generic_denoise_block(
                        self.l1_N,
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                        hparams.beta,
                        flatten=True,
                    )
                )
            elif layer_idx == len(hparams.layer_widths) - 1:
                lN_init_mean_func = nn.Linear(hparams.layer_widths[layer_idx - 1], hparams.layer_widths[layer_idx])
                lN_init_gen_func = nn.Identity()
                lN_init_funcs = {"inf": layer.MeanPlusNoise(hparams.layer_widths[-1], lN_init_mean_func, hparams.sigma_inf),
                                 "gen": layer.MeanPlusNoise(hparams.layer_widths[-1], lN_init_gen_func, hparams.sigma_gen)}
                lN_init_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
                lN_init = layer.IGFuncLayerWrapper(lN_init_funcs, lN_init_dists)
                layer_list.append(lN_init)
            else:
                layer_list.extend(
                    generic_denoise_block(
                        hparams.layer_widths[layer_idx - 1],
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                        hparams.beta,
                    )
                )
        # construct diffusion layers
        diffusion_nl = layer.DiffusionGenNL(hparams.layer_widths[-1], 1)
        
        lN_mean_func = layer.DiffusionInf(hparams.beta)
        lN_gen_func = layer.DiffusionGen(diffusion_nl, hparams.beta)
        lN_funcs = {"inf": layer.MeanPlusNoise(hparams.layer_widths[-1], lN_mean_func, np.sqrt(hparams.beta)),
                    "gen": layer.MeanPlusNoise(hparams.layer_widths[-1], lN_gen_func, hparams.sigma_gen)}
        lN_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
        diffusion_layer = layer.IGFuncLayerWrapper(lN_funcs, lN_dists)
        layer_list.append(diffusion_layer)
        layer_list.append(l_final)
        
        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        self.classifier = nn.Sequential(nn.Linear(hparams.layer_widths[0], 256), nn.Tanh(), nn.Linear(256,hparams.n_classes),  nn.LogSoftmax())