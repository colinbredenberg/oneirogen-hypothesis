from __future__ import annotations

import typing
from logging import getLogger as get_logger
from typing import Generic

from torch import Tensor, nn
from typing_extensions import TypeVar

from beyond_backprop.networks.layers import Sequential

from .layers import get_all_forward_activations

logger = get_logger(__name__)

# NOTE: Would be nice, but is a bit too difficult to make work atm, so it's not worth the extra
# code complexity.
# WithIndices = TypeVar("WithIndices", Literal[True], Literal[False], default=Literal[False])


class ConvBlock(Sequential):
    def __init__(
        self,
        conv: nn.Conv2d,
        rho: nn.Module,
        pool: nn.MaxPool2d,
    ):
        super().__init__(
            conv=conv,
            rho=rho,
            pool=pool,
        )
        self.conv: nn.Conv2d
        self.rho: nn.Module
        self.pool: nn.MaxPool2d


BlockType = TypeVar("BlockType", bound=Sequential)


class ConvArchitecture(Sequential, Generic[BlockType]):
    """Protocol that describes a network with a stack of blocks followed by fully-connected layers.

    Also includes a default forward pass implementation that can be reused or overwritten by
    subclasses.

    These architectures return either a single tensor, or a tuple of (logits, maxpool_indices),
    depending on the value of `return_maxpool_indices`.
    """

    _n_blocks: int
    return_maxpool_indices: bool = False

    @property
    def blocks(self) -> Sequential[BlockType]:
        """Convolutional blocks at the *start* of the network."""
        return self[: self._n_blocks]

    @property
    def fcs(self) -> Sequential:
        """Fully connected layers at the *end* of the network."""
        return self[self._n_blocks :]

    def forward(self, input: Tensor) -> Tensor | tuple[Tensor, list[Tensor]]:
        if not self.return_maxpool_indices:
            h = self.blocks(input)
            return self.fcs(h)

        max_indices: list[Tensor] = []

        h = input
        for block in self.blocks:
            h, block_max_indices = block(h)
            max_indices.append(block_max_indices)

        logits = self.fcs(h)
        return logits, max_indices

    if typing.TYPE_CHECKING:

        def __call__(self, input: Tensor) -> Tensor | tuple[Tensor, list[Tensor]]:
            ...


def get_all_forward_activations_and_indices(
    module: Sequential,
    x: Tensor,
    allow_grads_between_layers: bool = False,
) -> tuple[list[Tensor], list[Tensor]]:
    """Returns all the forward activations, as well as the maximum indices coming from the
    Maxpool2d layers of the forward network.

    NOTE: How we currently handle nn.MaxPool2d is annoying: We need to carry around the maxpool
    indices of the forward pass and use them in the backward (feedback) pass. If we used an
    AdaptiveAvgPool2d as the feedback equivalent of MaxPool2d for DTP, we wouldn't need to do this,
    but the feedback blocks would be more difficult to train.
    """

    if not (isinstance(module, ConvArchitecture) and module.return_maxpool_indices):
        return get_all_forward_activations(module, x, allow_grads_between_layers), []

    def _maybe_detach(v: Tensor) -> Tensor:
        return v if allow_grads_between_layers else v.detach()

    activations: list[Tensor] = []
    maxpool_indices: list[Tensor] = []
    for block in module.blocks:
        x, layer_maxpool_indices = block(_maybe_detach(x))
        activations.append(x)
        maxpool_indices.append(layer_maxpool_indices)
    for fc_layer in module.fcs:
        x = fc_layer(_maybe_detach(x))
        activations.append(x)
    return activations, maxpool_indices
