from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from torch import Tensor, nn
from typing_extensions import TypeGuard

ModuleType = TypeVar("ModuleType", bound=nn.Module)


@runtime_checkable
class Invertible(Protocol):
    """A Module that is "easy to invert" since it has known input and output shapes.

    It's easier to mark modules as invertible in-place than to create new subclass for every single
    nn.Module class that we want to potentially use in the forward net.
    """

    input_shape: tuple[int, ...] | None = None
    input_shapes: tuple[tuple[int, ...] | None, ...] = ()
    output_shape: tuple[int, ...] | None = None
    output_shapes: tuple[tuple[int, ...] | None, ...] = ()


@runtime_checkable
class FixedShapes(Invertible, Protocol):
    """A Module that has known input and output shapes which are potentially validated at
    runtime."""

    enforce_shapes: bool = False


def set_input_output_shapes_hook(
    module: Invertible,
    inputs: Tensor | tuple[Tensor, ...],
    outputs: Tensor | tuple[Tensor, ...],
) -> None:
    """Hook that sets the `input_shape` and `output_shape` attributes on the layers if not present.

    Also, if the `enforce_shapes` attribute is set to `True` on `module`, and the shapes don't
    match with their respective attributes, this will raise an error.
    """
    if isinstance(inputs, Tensor):
        input_shape = inputs.shape[1:]
        input_shapes = (input_shape,)
    elif isinstance(inputs, tuple):
        input_shapes = tuple(v.shape[1:] if isinstance(v, Tensor) else None for v in inputs)
        input_shape = input_shapes[0]
    else:
        return

    if isinstance(outputs, Tensor):
        output_shape = outputs.shape[1:]
        output_shapes = (output_shape,)
    elif isinstance(outputs, tuple):
        output_shapes = tuple(v.shape[1:] if isinstance(v, Tensor) else None for v in outputs)
        output_shape = output_shapes[0]
    else:
        return

    # Set the `input_shape`, `output_shape`, `enforce_shapes` attributes if not present:
    # NOTE: not using hasattr since some layers might have type annotations with empty tuple or smt.
    if getattr(module, "input_shape", ()) == ():
        module.input_shape = input_shape
    if getattr(module, "input_shapes", ()) == ():
        module.input_shapes = input_shapes

    if getattr(module, "output_shape", ()) == ():
        module.output_shape = output_shape
    if getattr(module, "output_shapes", ()) == ():
        module.output_shapes = output_shapes

    # NOTE: This isinstance check works with the `Invertible` procol since the attributes are there.
    assert isinstance(module, Invertible)

    # NOTE: This optional 'enforce_shapes' attribute is meant to make it easier to avoid issues,
    # by having the network be strict and consistent about the shapes it expects and produces.
    if isinstance(module, FixedShapes) and module.enforce_shapes:
        if input_shape != module.input_shape:
            raise RuntimeError(
                f"Layer {module} expected individual inputs to have shape {module.input_shape}, but "
                f"got {input_shape} "
            )
        if output_shape != module.output_shape:
            raise RuntimeError(
                f"Outputs of layer {module} have unexpected shape {output_shape} "
                f"(expected {module.output_shape})!"
            )


def is_invertible(network: nn.Module) -> TypeGuard[Invertible]:
    return bool(getattr(network, "input_shape", ()) and getattr(network, "output_shape", ()))


def set_input_output_shapes_on_forward(module: nn.Module) -> None:
    """Makes the module easier to "invert" by adding a hook to each layer that sets its
    `input_shape` and `output_shape` attributes during a forward pass.

    Modifies the module in-place.
    """
    if set_input_output_shapes_hook not in module._forward_hooks.values():
        module.register_forward_hook(set_input_output_shapes_hook)
    for child in module.modules():
        if set_input_output_shapes_hook not in child._forward_hooks.values():
            child.register_forward_hook(set_input_output_shapes_hook)
