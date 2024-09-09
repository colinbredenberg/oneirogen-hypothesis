from logging import getLogger as get_logger
from typing import Generic, Literal

from torch import Tensor, nn

from beyond_backprop.networks.conv_architecture import (
    BlockType,
    ConvArchitecture,
    ConvBlock,
)
from beyond_backprop.networks.layers import Sequential
from beyond_backprop.networks.lenet import LeNet
from beyond_backprop.networks.simple_vgg import SimpleVGG

from .pseudoinvert import (
    invert_conv2d,
    invert_maxpool2d,
    invert_sequential,
    pseudoinvert,
)

logger = get_logger(__name__)


class ConvFeedbackBlock(Sequential, Generic[BlockType]):
    def __init__(
        self,
        forward_net: BlockType,
    ):
        assert len(forward_net) == 3
        super().__init__(
            pool=invert_maxpool2d(forward_net[2]),
            rho=pseudoinvert(forward_net[1]),
            conv=invert_conv2d(forward_net[0]),
        )
        self.pool: nn.MaxUnpool2d
        self.rho: nn.Module
        self.conv: nn.ConvTranspose2d

    def forward(self, x: Tensor, max_indices: Tensor) -> Tensor:
        # Note: The first layer is the nn.MaxUnpool2d layer, which accepts the max indices as
        # input. We unfortunately can't just use the .forward of Sequential here, because it
        # doesn't seem to unpack the input tuple properly.
        h = self.pool(x, max_indices)
        h = self.rho(h)
        h = self.conv(h)
        return h


@pseudoinvert.register(ConvBlock)
def _invert_conv_block(forward_net: BlockType) -> ConvFeedbackBlock[BlockType]:
    return ConvFeedbackBlock(forward_net)


class ConvFeedbackNetwork(ConvArchitecture[ConvFeedbackBlock]):
    """Feedback network for the SimpleVGG and LeNet architectures."""

    return_maxpool_indices: Literal[False] = False

    # NOTE: This class could also have its own hparams if you wanted to.
    def __init__(self, forward_net: ConvArchitecture):
        self._n_blocks = len(forward_net.blocks)
        assert isinstance(forward_net, nn.Sequential)
        super().__init__(*invert_sequential(forward_net))

    @property
    def fcs(self) -> Sequential:
        """The fully connected layers at the *start* of the feedback network."""
        return self[: -self._n_blocks]

    @property
    def blocks(self) -> Sequential[ConvFeedbackBlock]:
        """The convolutional blocks at the *end* of the feedback network."""
        return self[-self._n_blocks :]

    def forward(self, x: Tensor, forward_maxpool_indices: list[Tensor]) -> Tensor:
        # NOTE: Assumes that the maxpool indices are given in the 'forward' order.
        feedback_maxpool_indices = forward_maxpool_indices[::-1]
        h = self.fcs(x)
        assert len(feedback_maxpool_indices) == len(self.blocks)
        for i, block in enumerate(self.blocks):
            logger.debug(f"{h.shape=}, {feedback_maxpool_indices[i].shape=}")
            h = block(h, feedback_maxpool_indices[i])
        return h


@pseudoinvert.register(SimpleVGG)
@pseudoinvert.register(LeNet)
def invert_conv_network(forward_net: ConvArchitecture) -> ConvFeedbackNetwork:
    return ConvFeedbackNetwork(forward_net)
