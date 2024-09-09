from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

from beyond_backprop.networks.conv_architecture import ConvArchitecture, ConvBlock
from beyond_backprop.networks.layers import Activation, Reshape, Sequential
from beyond_backprop.networks.network import ImageClassifierNetwork


class LeNetBlock(ConvBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        activation: type[nn.Module],
        return_maxpool_indices: bool = False,
    ):
        super().__init__(
            conv=nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=5,
                stride=1,
                padding=2,  # in Meuleman code padding=2
                bias=bias,
            ),
            rho=activation(),
            pool=nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, return_indices=return_maxpool_indices
            ),
        )


class LeNet(ConvArchitecture[LeNetBlock], ImageClassifierNetwork):
    @dataclass
    class HParams(ImageClassifierNetwork.HParams):
        channels: list[int] = field(default_factory=lambda: [32, 64])
        bias: bool = True

        activation: Activation = field(default=Activation.elu)
        """Choice of activation function to use."""

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: LeNet.HParams | None = None,
        return_maxpool_indices: bool = False,
    ):
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.hparams = hparams or self.HParams()
        self._return_maxpool_indices = return_maxpool_indices

        channels = [in_channels] + self.hparams.channels
        bias: bool = self.hparams.bias
        activation = self.hparams.activation.value
        self._n_blocks = len(self.hparams.channels)

        layers: list[nn.Module] = []
        # NOTE: Can use [0:] and [1:] below because zip will stop when the shortest
        # iterable is exhausted. This gives us the right number of blocks.
        for in_channels, out_channels in zip(channels[0:], channels[1:]):
            block = LeNetBlock(
                in_channels,
                out_channels,
                bias=bias,
                activation=activation,
                return_maxpool_indices=return_maxpool_indices,
            )
            layers.append(block)

        fc1 = Sequential(
            reshape=Reshape(target_shape=(-1,)),
            linear1=nn.LazyLinear(out_features=512, bias=bias),
            rho=activation(),
        )
        fc2 = Sequential(linear1=nn.Linear(in_features=512, out_features=n_classes, bias=bias))
        layers.append(fc1)
        layers.append(fc2)
        super().__init__(*layers)

    @property
    def return_maxpool_indices(self) -> bool:
        return self._return_maxpool_indices
