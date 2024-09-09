from __future__ import annotations

from torch import Tensor, nn
from torch.nn.common_types import _size_2_t

from beyond_backprop.networks.invertible import Invertible


class MaxUnpool2d(nn.MaxUnpool2d, Invertible):
    """Small tweak to nn.MaxUnpool2d so it uses the input shape from the forward pass net as its
    output shape when the `output_size` argument is not provided.

    This makes the networks easier to write and more flexible, since when the `output_size`
    argument isn't passed, the output shape of this layer can be different than the input shape of
    the forward net, depending on the kernel size, input size, and stride.
    """

    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: _size_2_t | None = None,
        padding: _size_2_t = 0,
    ):
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(
        self, input: Tensor, indices: Tensor, output_size: list[int] | None = None
    ) -> Tensor:
        if output_size is None and self.output_shape:
            output_size = list(self.output_shape[-2:])
        return super().forward(input=input, indices=indices, output_size=output_size)
