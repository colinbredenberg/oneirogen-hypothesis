from __future__ import annotations

import torch
from torch import nn
import numpy as np
import torch.distributions as dist
import torchvision.transforms.v2 as v2
from torch.distributions.normal import Normal
from torch.distributions.laplace import Laplace
from torch.distributions.log_normal import LogNormal
from torch.distributions.independent import Independent
from torch.distributions.one_hot_categorical import OneHotCategorical
import networkx as nx
import beyond_backprop.algorithms.common.layer as layer
from beyond_backprop.algorithms.common.graph_utils import sequential_graph
from .inf_gen_network import InfGenNetwork
from dataclasses import dataclass
from beyond_backprop.networks.network import Network
import pyrtools as pt
import plenoptic as po
from plenoptic.simulate import SteerablePyramidFreq
from plenoptic.synthesize import Eigendistortion
from plenoptic.tools.data import to_numpy

def generic_tanh_layer(in_dim, dim, out_dim, sigma_inf, sigma_gen, flatten=False):
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


def generic_meanscale_layer(in_dim, dim, out_dim, sigma_inf, sigma_gen, flatten=False):
    gen_mean_func = nn.Sequential(nn.Linear(out_dim, dim), nn.Tanh())
    gen_var_func = nn.Sequential(nn.Linear(out_dim, dim), nn.Sigmoid())
    if flatten:
        mean_func = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, dim), nn.Tanh())
        var_func = nn.Sequential(nn.Flatten(), nn.Linear(in_dim, dim), nn.Sigmoid())
    else:
        mean_func = nn.Sequential(nn.Linear(in_dim, dim), nn.Tanh())
        var_func = nn.Sequential(nn.Linear(in_dim, dim), nn.Sigmoid())
    funcs = {
        "inf": layer.MeanScale(mean_func, var_func, epsilon=0.001),
        "gen": layer.MeanScale(gen_mean_func, gen_var_func, epsilon=0.001),
    }
    dists = [
        lambda x: Independent(Normal(x[0], x[1]), 1),
        lambda x: Independent(dist.normal.Normal(x[0], x[1]), 1),
    ]
    l = layer.IGFuncLayerWrapper(funcs, dists)
    return l

def generic_meannoise_layer(in_dim, dim, out_dim, sigma_inf, sigma_gen, flatten=False, dropout=False, masking = False, batch_norm = True, dendrites = True):
    branch_num = 16
    gen_dendrite_nl = nn.Tanh() # nn.Softmax(dim=-1) #nn.LogSoftmax(dim = -1)#
    inf_dendrite_nl = nn.Tanh() # nn.Softmax(dim=-1) #nn.LogSoftmax(dim = -1)#nn.Tanh()#nn.Softmax(dim=-1)
    if dendrites:
        gen_mean_func_list = [layer.BranchedDendrite(out_dim, branch_num, dim, gen_dendrite_nl, batch_norm = batch_norm)]#
    else:
        gen_mean_func_list = [nn.Linear(out_dim, dim), nn.Tanh()]
        if batch_norm:
            gen_mean_func_list += [nn.BatchNorm1d(dim, affine = False)]
    # gen_var_func_list = [nn.Linear(out_dim, dim), nn.Sigmoid()]
    if dropout:
        # gen_mean_func_list = [nn.Dropout(p=0.5), *gen_mean_func_list]
        1+1
    gen_mean_func = nn.Sequential(*gen_mean_func_list)
    # gen_var_func = nn.Sequential(*gen_var_func_list)
    if dendrites:
        mean_func_list = [layer.BranchedDendrite(in_dim, branch_num, dim, inf_dendrite_nl, batch_norm = batch_norm)]
    else:
        mean_func_list = [nn.Linear(in_dim, dim), nn.Tanh()]
        if batch_norm:
             mean_func_list += [nn.BatchNorm1d(dim, affine = False)]
    # var_func_list = [nn.Linear(in_dim, dim), nn.Sigmoid()]
    if dropout:
        mean_func_list = [nn.Dropout(p=0.5), *mean_func_list]
    if masking:
        transform = v2.RandomErasing(p=0.5, scale = (1/8, 1/4), ratio = (0.3,3.3), value = 0)
        mean_func_list = [transform, *mean_func_list]
        # var_func_list = [nn.Dropout(p=0.5), *var_func_list]
    if flatten:
        mean_func_list = [nn.Flatten(), *mean_func_list]
        # var_func_list = [nn.Flatten(), *var_func_list]
    mean_func = nn.Sequential(*mean_func_list)
    # var_func = nn.Sequential(*var_func_list)
    funcs = {
        "inf": layer.MeanPlusNoise(dim, mean_func, sigma_inf),
        "gen": layer.MeanPlusNoise(dim, gen_mean_func, sigma_gen),
    }
    # funcs = {
    #     "inf": layer.MeanExpScale(mean_func, var_func),
    #     "gen": layer.MeanExpScale(gen_mean_func, gen_var_func),
    # }
    dists = [
        lambda x: Independent(Normal(x[0], x[1]), 1),
        lambda x: Independent(Normal(x[0], x[1]), 1),
    ]
    l = layer.IGFuncLayerWrapper(funcs, dists)
    return l

def generic_denoise_block(in_dim, dim, out_dim, sigma_inf, sigma_gen, beta, flatten = False):
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
    # l_denoise_nl = layer.DiffusionGenNL(dim, 1)
    l_denoise_mean_func = layer.DiffusionInf(beta) # layer.DiffusionGen(l_denoise_nl, beta)#
    l_denoise_funcs = {"inf": layer.MeanPlusNoise(dim, l_denoise_mean_func, np.sqrt(beta)),
                "gen": layer.MeanPlusNoise(dim, gen_mean_func, sigma_gen)}
    l_denoise_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
    l_denoise = layer.IGFuncLayerWrapper(l_denoise_funcs, l_denoise_dists)

    return [l, l_denoise]

def generic_conv_layer(shape, in_channels, channels, out_channels, kernel_size, padding, sigma_inf, sigma_gen):
    gen_mean_func_list = [nn.ConvTranspose2d(out_channels, channels, kernel_size, padding = padding), nn.Tanh()]
    gen_mean_func = nn.Sequential(*gen_mean_func_list)

    mean_func_list = [nn.Conv2d(in_channels, channels, kernel_size, padding = padding), nn.Tanh()]
    mean_func = nn.Sequential(*mean_func_list)

    funcs = {
        "inf": layer.ConvMeanPlusNoise([channels, *shape], mean_func, sigma_inf),
        "gen": layer.ConvMeanPlusNoise([channels, *shape], gen_mean_func, sigma_gen),
    }

    dists = [
        lambda x: Independent(Normal(x[0], x[1]), 3),
        lambda x: Independent(Normal(x[0], x[1]), 3),
    ]
    l = layer.IGFuncLayerWrapper(funcs, dists)
    return l

class WSLayeredModel(InfGenNetwork):
    """Network composed of 3 sequential IGFuncLayerWrapper layers for performing the Wake-Sleep
    algorithm on MNIST."""

    @dataclass
    class HParams(Network.HParams):
        l1_N: int = 0
        l2_N: int = 0
        l3_N: int = 0
        l4_N: int = 0
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: WSLayeredModel.HParams | None = None,
    ):
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        # declare the functions and distributions operating at each layer of the network
        l1_funcs = {
            "inf": nn.Identity(),
            "gen": nn.Sequential(
                nn.Unflatten(-1, [10, 24, 24]), nn.ConvTranspose2d(10, 1, 5), nn.Tanh()
            ),
        }  # "gen": nn.Sequential(nn.Linear(hparams.l2_N, hparams.l1_N), nn.Tanh())}
        l1_dists = [
            lambda x: Independent(Normal(x, hparams.sigma_inf), 1),
            lambda x: Independent(dist.normal.Normal(x, hparams.sigma_gen), 1),
        ]
        l2_funcs = {
            "inf": nn.Sequential(nn.Conv2d(1, 10, 5), nn.Flatten(), nn.ReLU()),
            "gen": nn.Sequential(nn.Linear(hparams.l3_N, 5760), nn.ReLU()),
        }
        l2_dists = [
            lambda x: Independent(Normal(x, hparams.sigma_inf), 1),
            lambda x: Independent(dist.normal.Normal(x, hparams.sigma_gen), 1),
        ]
        l4_mean_func = nn.Sequential(nn.Linear(hparams.l3_N, hparams.l4_N))
        l4_var_func = nn.Sequential(nn.Linear(hparams.l3_N, hparams.l4_N), nn.ReLU())
        l4_funcs = {
            "inf": layer.MeanScale(l4_mean_func, l4_var_func, epsilon=0.001),
            "gen": layer.MeanParams(hparams.l4_N, hparams.batch_size),
        }
        l4_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x, 1), 1),
        ]

        # l2_funcs = {"inf": nn.Sequential(nn.Linear(params['l1_N'], params['l2_N']), nn.ReLU()), "gen": lambda x: x}
        # l2_dists = [lambda x: dist.normal.Normal(x, params["sigma_inf"]), lambda x: dist.normal.Normal(torch.zeros(params["batch_size"], params["l2_N"]), 1)]

        # wrap the functions and distributions into their respective layers
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0])
        l2 = layer.IGFuncLayerWrapper(
            l2_funcs, l2_dists
        )  # generic_relu_layer(hparams.l1_N, hparams.l2_N, hparams.l3_N, hparams.sigma_inf, hparams.sigma_gen)
        l3 = generic_tanh_layer(
            5760, hparams.l3_N, hparams.l4_N, hparams.sigma_inf, hparams.sigma_gen
        )
        l4 = layer.IGFuncLayerWrapper(l4_funcs, l4_dists)

        # construct the generative and inference graphs
        graph = sequential_graph([l1, l2, l3, l4])
        gen_graph = sequential_graph([l4, l3, l2, l1])
        # graph = sequential_graph([l1, l2])
        # gen_graph = sequential_graph([l2,l1])

        # feed the graphs into the superclass initialization
        super().__init__(graphs=[graph, gen_graph])

        # assign each layer as a submodule of the network (it will be seen by the nn.Module functions)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4


class FCWSLayeredModel(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
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
        hparams: RMWSLayeredModel.HParams | None = None,
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
        # lN_funcs = {
        #     "inf": lN_mean_func,
        #     "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        # }
        # lN_dists = [
        #     lambda x: OneHotCategorical(logits=x),
        #     lambda x: OneHotCategorical(logits=x[0]),
        # ]
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
        # self.classifier = nn.Sequential(nn.Linear(hparams.layer_widths[2], 256), nn.Tanh(), nn.Linear(256,hparams.n_classes),  nn.LogSoftmax())
        self.classifier = nn.Sequential(nn.Linear(hparams.layer_widths[0], 256), nn.Tanh(), nn.Linear(256,hparams.n_classes),  nn.LogSoftmax())

        print(self.parameters())

class FCDiffusionModel(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128, 24)
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        diffusion_steps: int = 10
        beta: float = 0.1
        N_int: int = 128
        input_shape: tuple = (1, 28, 28)

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: RMWSLayeredModel.HParams | None = None,
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
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0])
        layer_list = [l1]

        l_final_mean_func = layer.DiffusionInf(hparams.beta)
        l_final_funcs = {"inf": layer.MeanPlusNoise(hparams.layer_widths[-1], l_final_mean_func, np.sqrt(hparams.beta)), "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size, 0, 1)}
        l_final_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]

        l_final = layer.IGFuncLayerWrapper(l_final_funcs, l_final_dists)

        #construct autoencoder layers
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
                layer_list.append(
                    generic_meannoise_layer(
                        hparams.layer_widths[layer_idx - 1],
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                    )
                )
        # construct diffusion layers
        diffusion_nl = layer.DiffusionGenNL(hparams.layer_widths[-1], hparams.diffusion_steps)
        
        for ii in range(0, hparams.diffusion_steps):
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
        print(self.parameters())

class LayerwiseDiffusionModel(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
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
        hparams: RMWSLayeredModel.HParams | None = None,
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
        print(self.parameters())

class FCRecurrentModel(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128, 24)
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        diffusion_steps: int = 10
        input_shape: tuple = (1, 28, 28)

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: RMWSLayeredModel.HParams | None = None,
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
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0])
        layer_list = [l1]

        gen_rec_layer = layer.RecurrentLayer(hparams.layer_widths[-1], hparams.diffusion_steps)
        inf_rec_layer = layer.RecurrentLayer(hparams.layer_widths[-1], hparams.diffusion_steps)

        l_final_mean_func = inf_rec_layer
        l_final_funcs = {"inf": layer.MeanPlusNoise(hparams.layer_widths[-1], l_final_mean_func, hparams.sigma_inf), "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size, 0, 1)}
        l_final_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]

        l_final = layer.IGFuncLayerWrapper(l_final_funcs, l_final_dists)

        #construct autoencoder layers
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
                    )
                )
            elif layer_idx == len(hparams.layer_widths) - 1:
                lN_init_mean_func = nn.Linear(hparams.layer_widths[layer_idx - 1], hparams.layer_widths[layer_idx])
                lN_init_gen_func = gen_rec_layer
                lN_init_funcs = {"inf": layer.MeanPlusNoise(hparams.layer_widths[-1], lN_init_mean_func, hparams.sigma_inf),
                                 "gen": layer.MeanPlusNoise(hparams.layer_widths[-1], lN_init_gen_func, hparams.sigma_gen)}
                lN_init_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
                lN_init = layer.IGFuncLayerWrapper(lN_init_funcs, lN_init_dists)
                layer_list.append(lN_init)
            else:
                layer_list.append(
                    generic_meannoise_layer(
                        hparams.layer_widths[layer_idx - 1],
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                    )
                )
        # construct diffusion layers
        
        
        for ii in range(0, hparams.diffusion_steps -1):
            lN_mean_func = inf_rec_layer
            lN_gen_func = gen_rec_layer
            lN_funcs = {"inf": layer.MeanPlusNoise(hparams.layer_widths[-1], lN_mean_func, hparams.sigma_inf),
                        "gen": layer.MeanPlusNoise(hparams.layer_widths[-1], lN_gen_func, hparams.sigma_gen)}
            lN_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
            diffusion_layer = layer.IGFuncLayerWrapper(lN_funcs, lN_dists)
            layer_list.append(diffusion_layer)
        layer_list.append(l_final)
        
        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())

class ConvDiffusionModel(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128, 24)
        kernel_size: int = 5
        recurrent_kernel_size: int = 5
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        diffusion_steps: int = 10
        beta: float = 0.1
        N_int: int = 128
        input_shape: tuple = (1, 28, 28)

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: RMWSLayeredModel.HParams | None = None,
    ):
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.l1_channels = self.hparams.input_shape[0]
        self.padding = int((self.hparams.kernel_size - 1)/2) #select padding to preserve the image shape
        self.recurrent_padding = int((self.hparams.recurrent_kernel_size - 1)/2)
        l1_gen_mean_func = nn.Sequential(
            nn.ConvTranspose2d(hparams.layer_widths[0], self.l1_channels, hparams.kernel_size, padding = self.padding),
            nn.Tanh()
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
        l_final_funcs = {"inf": layer.ConvMeanPlusNoise([hparams.layer_widths[-1], *hparams.input_shape[1::]], l_final_mean_func, np.sqrt(hparams.beta)), 
                         "gen": layer.ConvMeanParams([hparams.layer_widths[-1], *hparams.input_shape[1::]], hparams.batch_size, 0, 1)}
        l_final_dists = [lambda x: Independent(Normal(x[0], x[1]),3), lambda x: Independent(Normal(x[0],x[1]),3)]

        l_final = layer.IGFuncLayerWrapper(l_final_funcs, l_final_dists)

        #construct autoencoder layers
        for layer_idx, width in enumerate(hparams.layer_widths):
            if layer_idx == 0:
                layer_list.append(
                    generic_conv_layer(
                        hparams.input_shape[1::],
                        self.l1_channels,
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.kernel_size,
                        self.padding,
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                    )
                )
            elif layer_idx == len(hparams.layer_widths) - 1:
                lN_init_mean_func = nn.Conv2d(hparams.layer_widths[layer_idx - 1], hparams.layer_widths[layer_idx], hparams.kernel_size, padding = self.padding)
                lN_init_gen_func = nn.Identity()
                lN_init_funcs = {"inf": layer.ConvMeanPlusNoise([hparams.layer_widths[-1],*hparams.input_shape[1::]], lN_init_mean_func, hparams.sigma_inf),
                                 "gen": layer.ConvMeanPlusNoise([hparams.layer_widths[-1],*hparams.input_shape[1::]], lN_init_gen_func, hparams.sigma_gen)}
                lN_init_dists = [lambda x: Independent(Normal(x[0], x[1]),3), lambda x: Independent(Normal(x[0],x[1]),3)]
                lN_init = layer.IGFuncLayerWrapper(lN_init_funcs, lN_init_dists)
                layer_list.append(lN_init)
            else:
                layer_list.append(
                    generic_conv_layer(
                        hparams.input_shape[1::],
                        hparams.layer_widths[layer_idx - 1],
                        width,
                        hparams.layer_widths[layer_idx + 1],
                        hparams.kernel_size,
                        self.padding,
                        hparams.sigma_inf,
                        hparams.sigma_gen,
                    )
                )
        # construct diffusion layers
        lN_mean_func = layer.DiffusionInf(hparams.beta)
        lN_gen_func = layer.ConvDiffusionGen(hparams.layer_widths[-1], hparams.recurrent_kernel_size, self.recurrent_padding, hparams.diffusion_steps, hparams.beta)
        lN_funcs = {"inf": layer.ConvMeanPlusNoise([hparams.layer_widths[-1],*hparams.input_shape[1::]], lN_mean_func, np.sqrt(hparams.beta)),
                    "gen": layer.ConvMeanPlusNoise([hparams.layer_widths[-1],*hparams.input_shape[1::]], lN_gen_func, hparams.sigma_gen)}
        lN_dists = [lambda x: Independent(Normal(x[0], x[1]),3), lambda x: Independent(Normal(x[0],x[1]),3)]
        for ii in range(0, hparams.diffusion_steps):
            diffusion_layer = layer.IGFuncLayerWrapper(lN_funcs, lN_dists)
            layer_list.append(diffusion_layer)
        layer_list.append(l_final)
        
        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())

class MaskedFCWSLayeredModel(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128)
        l_N: int = 24
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: RMWSLayeredModel.HParams | None = None,
    ):
        hparams.layer_widths.append(hparams.l_N)
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.l1_N = int(np.prod(self.hparams.input_shape))
        l1_gen_mean_func = nn.Sequential(
            nn.Linear(hparams.layer_widths[0], self.l1_N),
            nn.Identity(),
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

        lN_mean_func = nn.Sequential(nn.Linear(hparams.layer_widths[-2], hparams.layer_widths[-1]))
        lN_var_func = nn.Sequential(nn.Linear(hparams.layer_widths[-2], hparams.layer_widths[-1]))
        # lN_funcs = {"inf": layer.MeanScale(lN_mean_func, lN_var_func, epsilon = 0.001), "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size, 0, 1)}
        # lN_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
        lN_funcs = {
            "inf": lN_mean_func,
            "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        }
        lN_funcs = {
            "inf": layer.MeanExpScale(lN_mean_func, lN_var_func),
            "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        }
        lN_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1),
        ]

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
                        dropout=True,
                        masking=False,
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
                        dropout=True,
                    )
                )

        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())

class SteerablePyramidModel(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128)
        l_N: int = 24
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)
        order: int = 3
        scales: int = 3

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: SteerablePyramidModel.HParams | None = None,
    ):
        hparams.layer_widths.append(hparams.l_N)
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        pyr_channel_num = 102 #((self.hparams.scales + 1) * self.hparams.order * 2) * self.hparams.input_shape[0]
        l0_shape = hparams.input_shape #[pyr_channel_num, *hparams.input_shape[1::]]
        l0_gen_mean_func = layer.InvSteerablePyramid(self.hparams.input_shape, self.hparams.order, self.hparams.scales, is_complex = True, is_polar = False)
        l0_funcs = {
                    "inf": layer.MeanPlusNoise(l0_shape, nn.Identity(), hparams.sigma_inf),
                    "gen": layer.MeanPlusNoise(l0_shape, l0_gen_mean_func, hparams.sigma_gen)
                    }
        l0_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]
        l0 = layer.IGFuncLayerWrapper(l0_funcs, l0_dists, idxs=[0])
        self.l1_N = int(np.prod(self.hparams.input_shape[1::])) * pyr_channel_num
        self.l1_shape = [pyr_channel_num, *hparams.input_shape[1::]]
        l1_mean_func = layer.SteerablePyramid(self.hparams.input_shape, self.hparams.order, self.hparams.scales, is_complex = True, is_polar = False)
        l1_gen_mean_func = nn.Sequential(
            nn.Linear(hparams.layer_widths[0], self.l1_N),
            nn.Identity(),
            nn.Unflatten(-1, self.l1_shape),
        )
        l1_funcs = {
            "inf": layer.MeanPlusNoise(self.l1_shape, l1_mean_func, hparams.sigma_inf),
            "gen": layer.MeanPlusNoise(self.l1_shape, l1_gen_mean_func, hparams.sigma_gen),
        }
        l1_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists)
        layer_list = [l0, l1]
        if len(hparams.layer_widths) == 1:
            layer_widths = [self.l1_N, *hparams.layer_widths]
            lN_mean_func = nn.Sequential(nn.Flatten(), nn.Linear(layer_widths[-2], hparams.layer_widths[-1]))
            lN_var_func = nn.Sequential(nn.Flatten(), nn.Linear(layer_widths[-2], hparams.layer_widths[-1]))
        else:
            layer_widths = hparams.layer_widths
            lN_mean_func = nn.Sequential(nn.Linear(layer_widths[-2], hparams.layer_widths[-1]))
            lN_var_func = nn.Sequential(nn.Linear(layer_widths[-2], hparams.layer_widths[-1]))

        lN_funcs = {
            "inf": layer.MeanExpScale(lN_mean_func, lN_var_func),
            "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        }
        lN_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1),
        ]

        lN = layer.IGFuncLayerWrapper(lN_funcs, lN_dists)
        if len(hparams.layer_widths) > 1:
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
                            dropout=True,
                            masking=False,
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
                            dropout=True,
                        )
                    )
        else:
            layer_list.append(lN)

        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())

class SteerablePyramidConvModel(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128)
        l_N: int = 24
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)
        order: int = 3
        scales: int = 3

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: SteerablePyramidConvModel.HParams | None = None,
    ):
        hparams.layer_widths.append(hparams.l_N)
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        pyr_channel_num = 162#((self.hparams.scales + 1) * self.hparams.order * 2) * self.hparams.input_shape[0] # 102 #132 #
        l0_shape = hparams.input_shape #[pyr_channel_num, *hparams.input_shape[1::]]
        l0_gen_mean_func = layer.InvSteerablePyramid(self.hparams.input_shape, self.hparams.order, self.hparams.scales, is_complex = True, is_polar = False)
        l0_funcs = {
                    "inf": layer.MeanPlusNoise(l0_shape, nn.Identity(), hparams.sigma_inf),
                    "gen": layer.MeanPlusNoise(l0_shape, l0_gen_mean_func, hparams.sigma_gen)
                    }
        l0_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]
        l0 = layer.IGFuncLayerWrapper(l0_funcs, l0_dists, idxs=[0])
        self.l1_N = int(np.prod(self.hparams.input_shape[1::])) * pyr_channel_num
        self.l1_shape = [pyr_channel_num, *hparams.input_shape[1::]]

        l2_channel_num = 10
        l2_kernel_size = 9
        l2_padding = 4
        l2_stride = 1

        l1_mean_func = layer.SteerablePyramid(self.hparams.input_shape, self.hparams.order, self.hparams.scales, is_complex = True, is_polar = False)
        l1_gen_mean_func = nn.Sequential(
            nn.ConvTranspose2d(l2_channel_num, pyr_channel_num, l2_kernel_size, l2_stride, l2_padding),
        )
        l1_funcs = {
            "inf": layer.MeanPlusNoise(self.l1_shape, l1_mean_func, hparams.sigma_inf),
            "gen": layer.MeanPlusNoise(self.l1_shape, l1_gen_mean_func, hparams.sigma_gen),
        }
        l1_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists)
        layer_list = [l0, l1]

        l2_shape = [l2_channel_num, 224, 224]#*hparams.input_shape[1::]]
        l2_N = np.prod(l2_shape)
        l2_mean_func = nn.Sequential(nn.Conv2d(pyr_channel_num, l2_channel_num, l2_kernel_size, l2_stride, l2_padding), nn.Tanh())
        l2_gen_mean_func = nn.Sequential(nn.Linear(hparams.layer_widths[-1], l2_N), nn.Tanh(), nn.Unflatten(-1,l2_shape))
        l2_funcs = {
            "inf": layer.MeanPlusNoise(l2_shape, l2_mean_func, hparams.sigma_inf),
            "gen": layer.MeanPlusNoise(l2_shape, l2_gen_mean_func, hparams.sigma_gen),#layer.ConvMeanParams(l2_shape, hparams.batch_size),#layer.MeanPlusNoise(l2_shape, l2_gen_mean_func, hparams.sigma_gen),
        }
        l2_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(Normal(x[0], x[1]), 3),#Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]
        l2 = layer.IGFuncLayerWrapper(l2_funcs, l2_dists)
        layer_list.append(l2)

        lN_mean_func = nn.Sequential(nn.Flatten(), nn.Linear(l2_N, hparams.layer_widths[-1]))
        lN_var_func = nn.Sequential(nn.Flatten(), nn.Linear(l2_N, hparams.layer_widths[-1]))

        lN_funcs = {
            "inf": layer.MeanPlusNoise(hparams.layer_widths[-1], lN_mean_func, hparams.sigma_inf),#"inf": layer.MeanExpScale(lN_mean_func, lN_var_func),
            "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        }
        lN_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1),
        ]

        lN = layer.IGFuncLayerWrapper(lN_funcs, lN_dists)
        
        layer_list.append(lN)

        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())

class TinyAutoencoderFCNet(InfGenNetwork):
    """Network composed of K sequential fully connected MLP layers of prespecified width for
    learning on natural images."""

    @dataclass
    class HParams(Network.HParams):
        layer_widths: tuple = (512, 128)
        l_N: int = 24
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (3, 224, 224)
        order: int = 3
        scales: int = 3

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: SteerablePyramidModel.HParams | None = None,
    ):
        hparams.layer_widths.append(hparams.l_N)
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()

        l0_shape = hparams.input_shape #[pyr_channel_num, *hparams.input_shape[1::]]
        l0_gen_mean_func = layer.TinyAutodecoder(requires_grad=False)

        l0_funcs = {
                    "inf": layer.MeanPlusNoise(l0_shape, nn.Identity(), hparams.sigma_inf),
                    "gen": layer.MeanPlusNoise(l0_shape, l0_gen_mean_func, hparams.sigma_gen)
                    }
        l0_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]
        l0 = layer.IGFuncLayerWrapper(l0_funcs, l0_dists, idxs=[0]) 
        self.l1_shape = [4, int(self.hparams.input_shape[1]/8), int(self.hparams.input_shape[2]/8)]
        self.l1_N = np.prod(self.l1_shape)
        l1_mean_func = layer.TinyAutoencoder(requires_grad=False)
        l1_gen_mean_func = nn.Sequential(
            nn.Linear(hparams.layer_widths[0], self.l1_N),
            nn.Identity(),
            nn.Unflatten(-1, self.l1_shape),
        )
        l1_funcs = {
            "inf": layer.MeanPlusNoise(self.l1_shape, l1_mean_func, hparams.sigma_inf),
            "gen": layer.MeanPlusNoise(self.l1_shape, l1_gen_mean_func, hparams.sigma_gen),
        }
        l1_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists)
        layer_list = [l0, l1]
        if len(hparams.layer_widths) == 1:
            layer_widths = [self.l1_N, *hparams.layer_widths]
            lN_mean_func = nn.Sequential(nn.Flatten(), nn.Linear(layer_widths[-2], hparams.layer_widths[-1]))
            lN_var_func = nn.Sequential(nn.Flatten(), nn.Linear(layer_widths[-2], hparams.layer_widths[-1]))
        else:
            layer_widths = hparams.layer_widths
            lN_mean_func = nn.Sequential(nn.Linear(layer_widths[-2], hparams.layer_widths[-1]))
            lN_var_func = nn.Sequential(nn.Linear(layer_widths[-2], hparams.layer_widths[-1]))

        lN_funcs = {
            "inf": layer.MeanExpScale(lN_mean_func, lN_var_func),
            "gen": layer.MeanParams(hparams.layer_widths[-1], hparams.batch_size),
        }
        lN_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1),
        ]

        lN = layer.IGFuncLayerWrapper(lN_funcs, lN_dists)
        if len(hparams.layer_widths) > 1:
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
                            dropout=True,
                            masking=False,
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
                            dropout=True,
                        )
                    )
        else:
            layer_list.append(lN)

        graph = sequential_graph(layer_list)
        gen_graph = sequential_graph(layer_list[::-1])

        super().__init__(graphs=[graph, gen_graph])
        print(self.parameters())

class RMWSLayeredModel(InfGenNetwork):
    """Network composed of 3 sequential IGFuncLayerWrapper layers for performing the Wake-Sleep
    algorithm on MNIST."""

    @dataclass
    class HParams(Network.HParams):
        l1_N: int = 0
        l2_N: int = 0
        l3_N: int = 0
        l4_N: int = 0
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: RMWSLayeredModel.HParams | None = None,
    ):
        self.n_classes = n_classes
        self.in_channels = in_channels
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        # declare the functions and distributions operating at each layer of the network
        l1_gen_mean_func = nn.Sequential(
            nn.Linear(hparams.l2_N, hparams.l1_N), nn.Tanh(), nn.Unflatten(-1, hparams.input_shape)
        )
        l1_funcs = {
            "inf": layer.MeanPlusNoise(hparams.input_shape, nn.Identity(), hparams.sigma_inf),
            "gen": layer.MeanPlusNoise(hparams.input_shape, l1_gen_mean_func, hparams.sigma_gen),
        }
        l1_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 3),
        ]

        l2_gen_mean_func = nn.Sequential(nn.Linear(hparams.l3_N, hparams.l2_N), nn.Tanh())
        l2_gen_var_func = nn.Sequential(nn.Linear(hparams.l3_N, hparams.l2_N), nn.Sigmoid())
        l2_mean_func = nn.Sequential(
            nn.Flatten(), nn.Linear(hparams.l1_N, hparams.l2_N), nn.Tanh()
        )
        l2_var_func = nn.Sequential(
            nn.Flatten(), nn.Linear(hparams.l1_N, hparams.l2_N), nn.Sigmoid()
        )
        l2_funcs = {
            "inf": layer.MeanScale(l2_mean_func, l2_var_func, epsilon=0.001),
            "gen": layer.MeanScale(l2_gen_mean_func, l2_gen_var_func, epsilon=0.001),
        }
        l2_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(dist.normal.Normal(x[0], x[1]), 1),
        ]

        l3_gen_mean_func = nn.Sequential(nn.Linear(hparams.l4_N, hparams.l3_N), nn.Tanh())
        l3_gen_var_func = nn.Sequential(nn.Linear(hparams.l4_N, hparams.l3_N), nn.Sigmoid())
        l3_mean_func = nn.Sequential(nn.Linear(hparams.l2_N, hparams.l3_N), nn.Tanh())
        l3_var_func = nn.Sequential(nn.Linear(hparams.l2_N, hparams.l3_N), nn.Sigmoid())
        l3_funcs = {
            "inf": layer.MeanScale(l3_mean_func, l3_var_func, epsilon=0.001),
            "gen": layer.MeanScale(l3_gen_mean_func, l3_gen_var_func, epsilon=0.001),
        }
        l3_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1),
        ]

        l4_mean_func = nn.Sequential(nn.Linear(hparams.l3_N, hparams.l4_N))
        l4_var_func = nn.Sequential(nn.Linear(hparams.l3_N, hparams.l4_N), nn.Sigmoid())
        l4_funcs = {
            "inf": layer.MeanScale(l4_mean_func, l4_var_func, epsilon=0.001),
            "gen": layer.MeanParams(hparams.l4_N, hparams.batch_size, 0, 1),
        }
        l4_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1),
        ]

        # l2_funcs = {"inf": nn.Sequential(nn.Linear(params['l1_N'], params['l2_N']), nn.ReLU()), "gen": lambda x: x}
        # l2_dists = [lambda x: dist.normal.Normal(x, params["sigma_inf"]), lambda x: dist.normal.Normal(torch.zeros(params["batch_size"], params["l2_N"]), 1)]

        # wrap the functions and distributions into their respective layers
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0])
        # l2 = generic_tanh_layer(hparams.l1_N, hparams.l2_N, hparams.l3_N, hparams.sigma_inf, hparams.sigma_gen)
        # l3 = generic_tanh_layer(hparams.l2_N, hparams.l3_N, hparams.l4_N, hparams.sigma_inf, hparams.sigma_gen)
        l2 = layer.IGFuncLayerWrapper(l2_funcs, l2_dists)
        l3 = layer.IGFuncLayerWrapper(l3_funcs, l3_dists)
        l4 = layer.IGFuncLayerWrapper(l4_funcs, l4_dists)

        # construct the generative and inference graphs
        graph = sequential_graph([l1, l2, l3, l4])
        gen_graph = sequential_graph([l4, l3, l2, l1])

        # feed the graphs into the superclass initialization
        super().__init__(graphs=[graph, gen_graph])

        # assign each layer as a submodule of the network (it will be seen by the nn.Module functions)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4


class PCAModel(InfGenNetwork):
    """Network composed of a single latent layer for doing Wake-Sleep on MNIST with a PCA-type
    model."""

    @dataclass
    class HParams(Network.HParams):
        l1_N: int = 0
        l2_N: int = 0
        l3_N: int = 0
        l4_N: int = 0
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: WSLayeredModel.HParams | None = None,
    ):
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        # declare the functions and distributions operating at each layer of the network

        l1_funcs = {"inf": layer.View([784]), "gen": nn.Linear(hparams.l2_N, hparams.l1_N)}
        l1_dists = [
            lambda x: Independent(Normal(x, hparams.sigma_inf), 1),
            lambda x: Independent(dist.normal.Normal(x, hparams.sigma_gen), 1),
        ]

        # wrap the functions and distributions into their respective layers
        l2_funcs = {
            "inf": nn.Linear(hparams.l1_N, hparams.l2_N),
            "gen": layer.MeanParams(hparams.l2_N, hparams.batch_size, mean_val=0.0),
        }
        l2_dists = [
            lambda x: Independent(Normal(x, hparams.sigma_inf), 1),
            lambda x: Independent(Normal(x, 1), 1),
        ]

        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0])
        l2 = layer.IGFuncLayerWrapper(l2_funcs, l2_dists)

        # construct the generative and inference graphs
        graph = sequential_graph([l1, l2])
        gen_graph = sequential_graph([l2, l1])
        # graph = sequential_graph([l1, l2])
        # gen_graph = sequential_graph([l2,l1])

        # feed the graphs into the superclass initialization
        super().__init__(graphs=[graph, gen_graph])

        # assign each layer as a submodule of the network (it will be seen by the nn.Module functions)
        self.l1 = l1
        self.l2 = l2


class SingleLayeredModel(InfGenNetwork):
    """Network composed of 3 sequential IGFuncLayerWrapper layers for performing the Wake-Sleep
    algorithm on MNIST."""

    @dataclass
    class HParams(Network.HParams):
        l1_N: int = 0
        l2_N: int = 0
        l3_N: int = 0
        l4_N: int = 0
        l_target_N: int = 0
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: WSLayeredModel.HParams | None = None,
    ):
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        # declare the functions and distributions operating at each layer of the network
        inf_network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hparams.l1_N, hparams.l2_N),
            nn.Tanh(),
            nn.Linear(hparams.l2_N, hparams.l3_N),
            nn.Tanh(),
        )
        gen_network = nn.Sequential(
            nn.Linear(hparams.l4_N, hparams.l3_N),
            nn.Tanh(),
            nn.Linear(hparams.l3_N, hparams.l2_N),
            nn.Tanh(),
            nn.Linear(hparams.l2_N, hparams.l1_N),
            nn.Tanh(),
            nn.Unflatten(-1, [3, 32, 32]),
        )
        # inf_network = nn.Sequential(nn.Conv2d(3,128,33, stride = 1),
        #                             nn.Tanh(),
        #                             nn.Conv2d(128,64,4, stride = 4),
        #                             nn.Tanh(),
        #                             nn.Conv2d(64,32, 4, stride = 4),
        #                             nn.Tanh(),
        #                             nn.Flatten(),
        #                             nn.Linear(4608, hparams.l3_N),
        #                             nn.Tanh(),
        #                             )
        # gen_network = nn.Sequential(nn.Linear(hparams.l4_N, hparams.l3_N),
        #                             nn.Tanh(),
        #                             nn.Linear(hparams.l3_N, 4608),
        #                             nn.Tanh(),
        #                             nn.Unflatten(-1, [32, 12, 12]),
        #                             nn.ConvTranspose2d(32, 64, 4, stride = 4),
        #                             nn.Tanh(),
        #                             nn.ConvTranspose2d(64, 128, 4, stride = 4),
        #                             nn.Tanh(),
        #                             nn.ConvTranspose2d(128,3,33, stride = 1),
        #                             nn.Identity(),
        #                             )
        # l1_funcs = {"inf": layer.MeanPlusNoise(784, layer.View([784]), hparams.sigma_inf), "gen": layer.MeanPlusNoise(784, gen_network, hparams.sigma_gen)}
        # l1_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(dist.normal.Normal(x[0], x[1]),1)]
        l1_funcs = {"inf": nn.Identity(), "gen": gen_network}
        l1_dists = [
            lambda x: Independent(Normal(x, hparams.sigma_inf), 1),
            lambda x: Independent(dist.normal.Normal(x, hparams.sigma_gen), 1),
        ]
        # l4_mean_func = nn.Sequential(inf_network, nn.Linear(hparams.l3_N, hparams.l4_N))
        # l4_funcs = {"inf": l4_mean_func, "gen": layer.MeanParams(hparams.l4_N, hparams.batch_size)}
        # l4_dists = [lambda x: OneHotCategorical(logits = x), lambda x: OneHotCategorical(logits = x)]

        # l2_funcs = {"inf": nn.Sequential(nn.Linear(params['l1_N'], params['l2_N']), nn.ReLU()), "gen": lambda x: x}
        # l2_dists = [lambda x: dist.normal.Normal(x, params["sigma_inf"]), lambda x: dist.normal.Normal(torch.zeros(params["batch_size"], params["l2_N"]), 1)]

        # wrap the functions and distributions into their respective layers
        l4_mean_func = nn.Sequential(inf_network, nn.Linear(hparams.l3_N, hparams.l4_N))
        l4_var_func = nn.Sequential(
            inf_network, nn.Linear(hparams.l3_N, hparams.l4_N), nn.Sigmoid()
        )
        l4_funcs = {
            "inf": layer.MeanScale(l4_mean_func, l4_var_func, epsilon=0.001),
            "gen": layer.MeanParams(hparams.l4_N, hparams.batch_size, mean_val=0.0),
        }
        l4_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 1),
            lambda x: Independent(Normal(x[0], x[1]), 1),
        ]

        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0])
        l4 = layer.IGFuncLayerWrapper(l4_funcs, l4_dists)

        # construct the generative and inference graphs
        # graph = sequential_graph([l1, l4])
        # gen_graph = sequential_graph([l4, l1])
        graph = nx.DiGraph()
        graph.add_edge(l1, l4)

        gen_graph = nx.DiGraph()
        gen_graph.add_edge(l4, l1)
        # graph = sequential_graph([l1, l2])
        # gen_graph = sequential_graph([l2,l1])

        # feed the graphs into the superclass initialization
        super().__init__(graphs=[graph, gen_graph])

        # assign each layer as a submodule of the network (it will be seen by the nn.Module functions)
        self.l1 = l1
        self.l4 = l4


class NormalizationModel(InfGenNetwork):
    """Network composed of 3 sequential IGFuncLayerWrapper layers for performing the Wake-Sleep
    algorithm on MNIST."""

    @dataclass
    class HParams(Network.HParams):
        l1_N: int = 0
        l2_N: int = 0
        l3_N: int = 0
        l4_N: int = 0
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1, 28, 28)

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: WSLayeredModel.HParams | None = None,
    ):
        # Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        self.n_classes = n_classes
        self.kernel_sizes = [9, 13, 21]
        self.num_filters = 512
        # inf_network = layer.ConvNormBlock(3,feature_num, norm_num, 32, 32, stride = 32, norm_stride = 32)
        # gen_network = layer.InvConvNormBlock(3,feature_num, norm_num, 32, 32, stride = 32, norm_stride = 32)

        # inf_network = nn.Sequential(layer.MultiScaleConvBlock(self.kernel_sizes, 3, self.num_filters),
        #                             nn.Tanh(),
        #                             nn.Conv2d(len(self.kernel_sizes)*self.num_filters, 200, 32, stride = 32, padding = 0))

        # gen_network = nn.Sequential(nn.ConvTranspose2d(200, len(self.kernel_sizes) * self.num_filters, 32, stride = 32, padding = 0),
        #                             nn.Tanh(),
        #                             layer.InvMultiScaleConvBlock(self.kernel_sizes, 3, self.num_filters))

        inf_network = nn.Conv2d(3, self.num_filters, 32, stride=32)
        gen_network = nn.ConvTranspose2d(self.num_filters, 3, 32, stride=32)
        # l_feature_size = [feature_num,7,7]
        # l_norm_size = [norm_num, 7, 7]

        l_feature_size = [128, self.num_filters, 1, 1]

        l1_funcs = {"inf": nn.Identity(), "gen": gen_network}
        l1_dists = [
            lambda x: Independent(Normal(x, hparams.sigma_inf), 3),
            lambda x: Independent(dist.normal.Normal(x, hparams.sigma_gen), 3),
        ]

        # wrap the functions and distributions into their respective layers
        l_feature_funcs = {
            "inf": layer.ConvMeanPlusNoise(l_feature_size, inf_network, hparams.sigma_inf),
            "gen": layer.ConvMeanParams(mean_val=0.0, scale=1.0),
        }
        l_feature_dists = [
            lambda x: Independent(Normal(x[0], x[1]), 3),
            lambda x: Normal(x[0], x[1]),
        ]

        # l_norm_funcs = {"inf": layer.ConvMeanPlusNoise(l_norm_size, inf_network, hparams.sigma_inf, idx = 1), "gen": layer.ConvMeanParams(l_norm_size, hparams.batch_size)}
        # l_norm_dists = [lambda x: Independent(Normal(x[0], x[1]),3), lambda x: Independent(Normal(x[0],x[1]),3)]

        # l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs = [0], gen_stack = False)
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs=[0])
        l_feature = layer.IGFuncLayerWrapper(
            l_feature_funcs, l_feature_dists, batch_num=[[], l_feature_size]
        )
        # l_norm = layer.IGFuncLayerWrapper(l_norm_funcs, l_norm_dists)

        # construct the generative and inference graphs
        graph = nx.DiGraph()
        graph.add_edge(l1, l_feature)
        # graph.add_edge(l1, l_norm)

        gen_graph = nx.DiGraph()
        gen_graph.add_edge(l_feature, l1)
        # gen_graph.add_edge(l_norm, l1)
        # graph = sequential_graph([l1, l_feature])
        # gen_graph = sequential_graph([l_feature, l1])
        # feed the graphs into the superclass initialization
        super().__init__(graphs=[graph, gen_graph])

        # assign each layer as a submodule of the network (it will be seen by the nn.Module functions)
        self.l1 = l1
        # self.l_norm = l_norm
        self.l_feature = l_feature
