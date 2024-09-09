from __future__ import annotations

import enum
from collections import OrderedDict
from typing import Iterator, Sequence, overload

from torch import Tensor, nn
from torch.nn.modules.container import _copy_to_script_wrapper

from beyond_backprop.utils.types import ModuleType

from .invertible import Invertible


class Activation(enum.Enum):
    relu = nn.ReLU
    elu = nn.ELU


# Small typing fixes for torch.nn.Sequential
# TODO: Could write a Sequential that consumes the maxpool indices as needed!
class Sequential(nn.Sequential, Sequence[ModuleType]):
    @overload
    def __init__(self, *args: ModuleType) -> None:
        ...

    @overload
    def __init__(self, **kwargs: ModuleType) -> None:
        ...

    @overload
    def __init__(self, arg: dict[str, ModuleType]) -> None:
        ...

    def __init__(self, *args, **kwargs):
        if kwargs:
            assert not args, "can only use *args or **kwargs, not both"
            args = (OrderedDict(kwargs),)
        super().__init__(*args)

    @overload
    def __getitem__(self, idx: int) -> ModuleType:
        ...

    @overload
    def __getitem__(self, idx: slice) -> Sequential[ModuleType]:
        ...

    @_copy_to_script_wrapper
    def __getitem__(self, idx: int | slice) -> Sequential | ModuleType:
        if isinstance(idx, slice):
            # NOTE: Fixing this here, subclass constructors shouldn't be called on getitem with
            # slice.
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __iter__(self) -> Iterator[ModuleType]:
        return super().__iter__()  # type: ignore

    # Violates the LSP, but eh.
    def __setitem__(self, idx: int, module: ModuleType) -> None:
        return super().__setitem__(idx, module)


def forward_each(module: nn.Sequential, xs: list[Tensor]) -> list[Tensor]:
    """Gets the outputs of every layer, given inputs for each layer `xs`.

    Parameters
    ----------
    x : List[Tensor]
        A list of tensors, one per layer, which will be used as the inputs for each
        forward layer.

    Returns
    -------
    List[Tensor]
        The outputs of each forward layer.
    """
    xs = list(xs) if not isinstance(xs, (list, tuple)) else xs
    assert len(xs) == len(module)
    return [layer(x_i) for layer, x_i in zip(module, xs)]


def get_all_forward_activations(
    module: nn.Sequential,
    x: Tensor,
    allow_grads_between_layers: bool = False,
) -> list[Tensor]:
    """Gets the outputs of all forward layers for the given input.

    Parameters
    ----------
    x : Tensor
        Input tensor.

    allow_grads_between_layers : bool, optional
        Whether to allow gradients to flow from one layer to the next.
        When `False` (default), outputs of each layer are detached before being
        fed to the next layer.

    Returns
    -------
    List[Tensor]
        The outputs of each forward layer.
    """
    activations: list[Tensor] = []
    for layer in module:
        x = layer(x if allow_grads_between_layers else x.detach())
        activations.append(x)
    return activations


class Reshape(nn.Module, Invertible):
    def __init__(self, target_shape: tuple[int, ...]):
        super().__init__()
        self.target_shape = tuple(target_shape)
        # add_hooks(self)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 1:
            return inputs.reshape(self.target_shape)
        outputs = inputs.reshape([inputs.shape[0], *self.target_shape])
        if self.target_shape == (-1,):
            self.target_shape = outputs.shape[1:]
        if not self.input_shape:
            self.input_shape = inputs.shape[1:]
        if not self.output_shape:
            self.output_shape = outputs.shape[1:]
        return outputs

    def extra_repr(self) -> str:
        return f"({self.input_shape} -> {self.target_shape})"

    # def __repr__(self):
    #     return f"{type(self).__name__}({self.input_shape} -> {self.target_shape})"
