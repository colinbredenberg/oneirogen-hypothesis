from __future__ import annotations

import beyond_backprop.algorithms.common.layer as layer
from beyond_backprop.algorithms.common.layered_network import LayeredNetwork
from beyond_backprop.algorithms.common.graph_utils import sequential_graph
import torch
from torch import nn
import torch.nn.functional as F
from beyond_backprop.networks.network import Network
from dataclasses import dataclass, field

class FALayeredModel(LayeredNetwork):
    """
    Constructs a model with an architecture identical to the Model class
    for use with the FeedbackAlignment algorithm
    """
    @dataclass
    class HParams(Network.HParams):
        middle_layer_width: int = 50
    
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: FALayeredModel.HParams | None = None,
    ):
        self.hparams = hparams or self.HParams()
        l0 = layer.FuncLayerWrapper(layer.FAConv2d(1, 10, kernel_size=5), lambda x: F.relu(F.max_pool2d(x, 2)))
        l1 = layer.FuncLayerWrapper(layer.FAConv2d(10, 20, kernel_size=5),
                                    lambda x: torch.reshape(F.relu(F.max_pool2d(x, 2)), [-1, 320]))
        l2 = layer.FuncLayerWrapper(layer.FALinear(320, hparams.middle_layer_width), F.relu)
        l3 = layer.FuncLayerWrapper(layer.FALinear(hparams.middle_layer_width, n_classes), lambda x: F.log_softmax(x, -1))

        graph = sequential_graph([l0, l1, l2, l3])
        super().__init__(graph=graph)

        self.l0 = l0
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3