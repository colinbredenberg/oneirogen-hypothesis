from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

from beyond_backprop.networks.layers import Activation, Reshape, Sequential

from .conv_architecture import ConvArchitecture, ConvBlock
from .network import ImageClassifierNetwork


class SimpleVGGBlock(ConvBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool,
        activation: type[nn.Module],
        return_indices: bool = False,
    ):
        super().__init__(
            conv=nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            rho=activation(),
            pool=nn.MaxPool2d(kernel_size=2, stride=2, return_indices=return_indices),
        )


class SimpleVGG(ConvArchitecture[SimpleVGGBlock], ImageClassifierNetwork):
    @dataclass
    class HParams(ImageClassifierNetwork.HParams):
        channels: list[int] = field(default_factory=[128, 128, 256, 256, 512].copy)
        bias: bool = True

        activation: Activation = Activation.elu
        """Choice of activation function to use."""

    # TODO: Potentially Use nn.LazyConv2d to remove the need to know exactly how many input
    # channels there are?
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        hparams: SimpleVGG.HParams | None = None,
        return_maxpool_indices: bool = False,
    ):
        self.in_channels = in_channels
        self.n_classes = n_classes
        self._return_maxpool_indices = return_maxpool_indices
        self.hparams = hparams or self.HParams()

        activation: type[nn.Module] = self.hparams.activation.value
        channels = [in_channels] + self.hparams.channels
        self._n_blocks = len(self.hparams.channels)

        # NOTE: Can use [0:] and [1:] below because zip will stop when the shortest
        # iterable is exhausted. This gives us the right number of blocks.
        layers: list[nn.Module] = []
        for in_channels, out_channels in zip(channels[0:], channels[1:]):
            block = SimpleVGGBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                bias=self.hparams.bias,
                activation=activation,
                return_indices=return_maxpool_indices,
            )
            layers.append(block)
        fc = Sequential(
            reshape=Reshape(target_shape=(-1,)),
            linear=nn.LazyLinear(out_features=n_classes, bias=self.hparams.bias),
        )
        layers.append(fc)
        super().__init__(*layers)

    @property
    def fc(self) -> Sequential:
        """The fully connected layer at the end of the network."""
        return self[-1]

    @property
    def return_maxpool_indices(self) -> bool:
        return self._return_maxpool_indices
