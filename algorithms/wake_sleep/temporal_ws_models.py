from __future__ import annotations

import torch
from torch import nn
import numpy as np
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
from .inf_gen_network import InfGenNetwork
from .inf_gen_network import TemporalPlaceHolder
from dataclasses import dataclass, field
from beyond_backprop.networks.network import Network

class TemporalLayeredModel(InfGenNetwork):
    """Network composed of 3 sequential IGFuncLayerWrapper layers for performing the Wake-Sleep algorithm on MNIST"""
    @dataclass
    class HParams(Network.HParams):
        l1_N: int = 0
        l2_N: int = 0
        l3_N: int = 0
        l4_N: int = 0
        sigma_inf: float = 0
        sigma_gen: float = 0
        batch_size: int = 0
        input_shape: tuple = (1,28,28)

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: TemporalLayeredModel.HParams | None = None,
    ):
        self.n_classes = n_classes
        self.in_channels = in_channels
        #Graph nodes have to be assigned before super().__init__ call
        self.hparams = hparams or self.HParams()
        #declare the functions and distributions operating at each layer of the network
        l1_gen_mean_func = nn.Sequential(nn.Linear(hparams.l2_N, hparams.l1_N), nn.Tanh(), nn.Unflatten(-1, hparams.input_shape))
        l1_funcs = {"inf": layer.MeanPlusNoise(hparams.input_shape, nn.Identity(), hparams.sigma_inf), "gen": layer.MeanPlusNoise(hparams.input_shape, l1_gen_mean_func, hparams.sigma_gen)}
        l1_dists = [lambda x: Independent(Normal(x[0], x[1]),3), lambda x: Independent(dist.normal.Normal(x[0], x[1]),3)]

        l2_gen_mean_func = nn.Sequential(nn.Linear(hparams.l3_N, hparams.l2_N), nn.Tanh())
        # l2_gen_var_func = nn.Sequential(nn.Linear(hparams.l3_N, hparams.l2_N), nn.Sigmoid())
        l2_mean_func = nn.Sequential(nn.Linear(hparams.l1_N + hparams.l4_N, hparams.l2_N), nn.Tanh())
        # l2_var_func = nn.Sequential(nn.Linear(hparams.l1_N + hparams.l4_N, hparams.l2_N), nn.Sigmoid())
        # l2_funcs = {"inf": layer.MeanScale(l2_mean_func, l2_var_func, epsilon = 0.001), "gen":layer.MeanScale(l2_gen_mean_func,l2_gen_var_func, epsilon = 0.001)}
        l2_funcs = {"inf": layer.MeanPlusNoise(hparams.l2_N, l2_mean_func, hparams.sigma_inf), "gen":layer.MeanPlusNoise(hparams.l2_N, l2_gen_mean_func, hparams.sigma_gen)}

        l2_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(dist.normal.Normal(x[0], x[1]),1)]

        l3_gen_mean_func = nn.Sequential(nn.Linear(hparams.l4_N, hparams.l3_N), nn.Tanh())
        # l3_gen_var_func =  nn.Sequential(nn.Linear(hparams.l4_N, hparams.l3_N), nn.Sigmoid())
        l3_mean_func = nn.Sequential(nn.Linear(hparams.l2_N + hparams.l4_N, hparams.l3_N), nn.Tanh())
        # l3_var_func = nn.Sequential(nn.Linear(hparams.l2_N + hparams.l4_N, hparams.l3_N), nn.Sigmoid())
        # l3_funcs = {"inf": layer.MeanScale(l3_mean_func, l3_var_func, epsilon = 0.001), "gen": layer.MeanScale(l3_gen_mean_func,l3_gen_var_func, epsilon = 0.001)}
        l3_funcs = {"inf": layer.MeanPlusNoise(hparams.l3_N, l3_mean_func, hparams.sigma_inf), "gen":layer.MeanPlusNoise(hparams.l3_N, l3_gen_mean_func, hparams.sigma_gen)}

        l3_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]

        l4_gen_mean_func = nn.Sequential(nn.Linear(hparams.l4_N, hparams.l4_N), nn.Tanh())
        # l4_gen_var_func =  nn.Sequential(nn.Linear(hparams.l4_N, hparams.l4_N), nn.Sigmoid())
        l4_mean_func = nn.Sequential(nn.Linear(hparams.l3_N + self.n_classes, hparams.l4_N))
        # l4_var_func = nn.Sequential(nn.Linear(hparams.l3_N + self.n_classes, hparams.l4_N), nn.Sigmoid())
        # l4_total_func = layer.MeanScale(l4_mean_func, l4_var_func, epsilon = 0.001)
        
        # l4_funcs = {"inf": l4_total_func, "gen": layer.MeanParams(hparams.l4_N, hparams.batch_size, 0, 1)}
        # l4_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]
        # l4_funcs = {"inf": l4_mean_func, "gen": layer.MeanParams(hparams.l4_N, hparams.batch_size, 0,1)}
        l4_funcs = {"inf": layer.MeanPlusNoise(hparams.l4_N, l4_mean_func, hparams.sigma_inf), "gen": layer.MeanParams(hparams.l4_N, hparams.batch_size, 0,1)}

        l4_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0], 1),1)]

        # l4_trans_funcs = {"inf": l4_total_func, "gen": layer.MeanScale(l4_gen_mean_func,l4_gen_var_func, epsilon = 0.001)}
        l4_trans_funcs = {"inf": layer.MeanPlusNoise(hparams.l4_N, l4_mean_func, hparams.sigma_inf), "gen":layer.MeanPlusNoise(hparams.l4_N, l4_gen_mean_func, hparams.sigma_gen)}

        l4_trans_dists = [lambda x: Independent(Normal(x[0], x[1]),1), lambda x: Independent(Normal(x[0],x[1]),1)]

        l_label_funcs = {"inf": nn.Identity(), "gen": nn.Linear(hparams.l4_N, self.n_classes)}
        l_label_dists = [lambda x: OneHotCategorical(logits = x), lambda x: OneHotCategorical(logits = x)]
        #wrap the functions and distributions into their respective layers
        l1 = layer.IGFuncLayerWrapper(l1_funcs, l1_dists, idxs = [0], shape = torch.Size([hparams.batch_size, hparams.l1_N]))
        l2 = layer.IGFuncLayerWrapper(l2_funcs, l2_dists, shape = torch.Size([hparams.batch_size, hparams.l2_N]), flatten= True)
        l3 = layer.IGFuncLayerWrapper(l3_funcs, l3_dists, shape = torch.Size([hparams.batch_size, hparams.l3_N]))
        l4 = layer.IGFuncLayerWrapper(l4_funcs, l4_dists, shape = torch.Size([hparams.batch_size, hparams.l4_N]), is_t0_layer = True, is_transition_layer = False)
        l4_trans = layer.IGFuncLayerWrapper(l4_trans_funcs, l4_trans_dists, shape = torch.Size([hparams.batch_size, hparams.l4_N]), is_t0_layer = False, is_transition_layer = True)
        l_label = layer.IGFuncLayerWrapper(l_label_funcs, l_label_dists, idxs = [1], shape = torch.Size([hparams.batch_size, self.n_classes]))
        #construct the generative and inference graphs
        graph = sequential_graph([l1, l2, l3, l4_trans])
        graph.add_edge(l_label, l4_trans)
        gen_graph = sequential_graph([l4_trans, l3,l2,l1])
        gen_graph.add_edge(l4_trans, l_label)
        # time_graph = sequential_graph([l4, l3, l2])
        # time_graph.add_node(l1)
        # time_gen_graph = nx.DiGraph()
        # time_gen_graph.add_edge(l4, l4)
        # for node in [l1, l2, l3]:
        #     time_gen_graph.add_node(node)
        time_placeholders = [TemporalPlaceHolder(l.shape) for l in [l1, l2, l3, l4_trans, l_label]]
        time_assoc = [[l1, time_placeholders[0]], [l2, time_placeholders[1]], [l3, time_placeholders[2]], [l4_trans, time_placeholders[3]], [l4, time_placeholders[3]], [l_label, time_placeholders[4]]]
        graph.add_edge(time_placeholders[3], l3)
        graph.add_edge(time_placeholders[3], l2)
        # graph.add_edge(time_placeholders[2], l2)
        gen_graph.add_edge(time_placeholders[3], l4_trans)
        for l in time_placeholders:
            graph.add_node(l)
            gen_graph.add_node(l)

        t0_graph = sequential_graph([l1, l2, l3, l4])
        t0_graph.add_edge(l_label, l4)
        t0_graph.add_edge(time_placeholders[3], l3)
        t0_graph.add_edge(time_placeholders[3], l2)
        gen_t0_graph = sequential_graph([l4, l3, l2, l1])
        gen_t0_graph.add_edge(l4, l_label)
        for l in time_placeholders:
            t0_graph.add_node(l)
            gen_t0_graph.add_node(l)

        #feed the graphs into the superclass initialization
        super().__init__(graphs = [graph, gen_graph], t0_graphs = [t0_graph, gen_t0_graph], time_assoc = time_assoc)

        #assign each layer as a submodule of the network (it will be seen by the nn.Module functions)
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l4_trans = l4_trans
        self.l_label = l_label