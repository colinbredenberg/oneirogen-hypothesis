"""Function used to create the feedback network given the forward-pass network.

This gets used by the DTP algorithm and its variants, but other algorithms are also free to use it.
"""
from __future__ import annotations

from functools import singledispatch
from typing import NoReturn, OrderedDict, TypeVar
import warnings

from torch import nn

from beyond_backprop.networks.invertible import Invertible, is_invertible
from beyond_backprop.networks.layers import Reshape

from .feedback_layers import MaxUnpool2d

ModuleType = TypeVar("ModuleType", bound=nn.Module)


@singledispatch
def pseudoinvert(network: nn.Module | Invertible) -> nn.Module:
    """Creates a network that performs the pseudo-inverse operation of `network`.

    NOTE: All concrete handlers below usually assume that a layer has been marked as 'Invertible',
    meaning that it has the `input_shape` and `output_shape` attributes set, from having been used
    in at least one forward pass.

    Parameters
    ----------
    layer : nn.Module
        Layer of the forward-pass.

    Returns
    -------
    nn.Module
        Layer to use at the same index as `layer` in the backward-pass model.

    Raises
    ------
    NotImplementedError
        When we don't know what type of layer to use for the backward pass of `layer`.
    """
    raise NotImplementedError(f"Don't know how to create the feedback network for {network}!")


@pseudoinvert.register
def invert_linear(layer: nn.Linear) -> nn.Linear:
    # NOTE: Not sure how to handle the bias term.
    backward = type(layer)(
        in_features=layer.out_features,
        out_features=layer.in_features,
        bias=layer.bias is not None,
    )
    return backward


@pseudoinvert.register
def _invert_lazylinear(layer: nn.LazyLinear) -> NoReturn:
    raise RuntimeError(
        f"Can't invert LazyLinear layer {layer}. You need to use it in a forward pass first."
    )


@pseudoinvert.register
def invert_dropout(layer: nn.Dropout) -> nn.Dropout:
    warnings.warn(
        RuntimeWarning(
            "Returning a Dropout layer as the pseudoinverse of a Dropout layer. "
            "Not sure if that makes any sense in terms of biological-plausibility."
        )
    )
    return nn.Dropout(p=layer.p)


@pseudoinvert.register(nn.Sequential)
def invert_sequential(module: nn.Sequential) -> nn.Sequential:
    """Returns a Module that can be used to compute or approximate the inverse operation of `self`.

    NOTE: In the case of Sequential, the order of the layers in the returned network
    is reversed compared to the input.
    """
    # assert module.input_shape and module.output_shape, "Use the net before inverting."
    # NOTE: Inverting a ResNet (which subclasses Sequential) doesn't try to create another ResNet!
    # It just returns a Sequential.
    return nn.Sequential(
        OrderedDict(
            (name, pseudoinvert(module))
            for name, module in reversed(list(module._modules.items()))
        ),
    )


@pseudoinvert.register(nn.Identity)
def invert_identity(module: nn.Identity) -> nn.Identity:
    return nn.Identity()


@pseudoinvert.register
def invert_conv2d(layer: nn.Conv2d) -> nn.ConvTranspose2d:
    assert len(layer.kernel_size) == 2
    assert len(layer.stride) == 2
    assert len(layer.padding) == 2
    assert len(layer.output_padding) == 2
    k_h, k_w = layer.kernel_size
    s_h, s_w = layer.stride
    assert not isinstance(layer.padding, str)
    p_h, p_w = layer.padding
    d_h, d_w = layer.dilation
    op_h, op_w = layer.output_padding
    assert k_h == k_w, "only support square kernels for now"
    assert s_h == s_w, "only support square stride for now"
    assert p_h == p_w, "only support square padding for now"
    assert d_h == d_w, "only support square padding for now"
    assert op_h == op_w, "only support square output_padding for now"

    backward = nn.ConvTranspose2d(
        in_channels=layer.out_channels,
        out_channels=layer.in_channels,
        kernel_size=(k_h, k_w),
        # TODO: Not 100% sure about these values:
        stride=(s_h, s_w),
        dilation=d_h,
        padding=(p_h, p_w),
        # TODO: Get this value programmatically.
        output_padding=(s_h - 1, s_w - 1),
        bias=layer.bias is not None,
        # output_padding=(op_h + 1, op_w + 1),  # Not sure this will always hold
    )
    return backward


@pseudoinvert.register(nn.ReLU)
def invert_relu(activation_layer: nn.Module) -> nn.Module:
    # Note: ReLU isn't invertible, but this is actually usually called for the last layer in a
    # block, and it makes sense in that case for the last layer of the feedback block to be a relu
    # as well.
    # TODO: Double-Check that this makes sense.
    return nn.ReLU(inplace=False)


@pseudoinvert.register(nn.ELU)
def _invert_elu(activation_layer: nn.ELU) -> nn.Module:
    return nn.ELU(alpha=activation_layer.alpha, inplace=False)


@pseudoinvert.register(nn.AdaptiveAvgPool2d)
@pseudoinvert.register(nn.AvgPool2d)
def invert_adaptive_avgpool2d(module: nn.AvgPool2d) -> nn.AdaptiveAvgPool2d:
    """Returns a nn.AdaptiveAvgPool2d, which will actually upsample the input!"""
    assert module.input_shape and module.output_shape, "Use the net before inverting."
    # TODO: Look into using Upsample rather than AdaptiveAvgPool2d, since it might give back an
    # output that is more like the input, e.g. using nearest neighbor interpolation.
    # return nn.Upsample(size=module.input_shape[-2:],)
    return nn.AdaptiveAvgPool2d(
        output_size=module.input_shape[-2:],  # type: ignore
    )


@pseudoinvert.register
def invert_reshape(module: Reshape) -> Reshape:
    assert module.input_shape and module.output_shape, "Use the net before inverting."
    layer = type(module)(target_shape=module.input_shape)
    layer.output_shape = module.input_shape
    layer.input_shape = module.output_shape
    return layer


@pseudoinvert.register
def invert_flatten(module: nn.Flatten) -> Reshape:
    assert is_invertible(module), "Use the net before inverting."
    assert isinstance(
        module, Invertible
    ), "Mark the net as invertible and use it in a forward pass first!"
    assert module.input_shape is not None
    layer = Reshape(target_shape=module.input_shape)
    layer.output_shape = module.input_shape
    layer.input_shape = module.output_shape
    return layer


@pseudoinvert.register
def invert_maxunpool2d(module: nn.AdaptiveMaxPool2d) -> nn.AdaptiveMaxPool2d:
    raise NotImplementedError("Never really need to invert a max Unpool layer.")
    assert module.input_shape and module.output_shape
    assert len(module.input_shape) > 2
    return nn.AdaptiveMaxPool2d(
        output_size=module.input_shape[-2:],
    )


@pseudoinvert.register
def invert_maxpool2d(module: nn.MaxPool2d) -> MaxUnpool2d:
    assert isinstance(module, Invertible), "mark the network as invertible first!"
    assert module.input_shape and module.output_shape, "Use the net before inverting."
    m = MaxUnpool2d(
        kernel_size=module.kernel_size,
        stride=module.stride,  # todo: double-check that this is correct
        padding=module.padding,  # todo: double-check that this is correct
    )
    m.output_shape = module.input_shape
    m.input_shape = module.output_shape
    return m


@pseudoinvert.register(nn.BatchNorm2d)
def invert_batchnorm(layer: nn.BatchNorm2d) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(
        num_features=layer.num_features,
        eps=layer.eps,
        momentum=layer.momentum,
        affine=layer.affine,
        track_running_stats=layer.track_running_stats,
    )


# @invert.register(nn.BatchNorm2d)
# def invert_batchnorm(
#     layer: nn.BatchNorm2d, init_symetric_weights: bool = False
# ) -> BatchUnNormalize:
#     # TODO: Is there a way to initialize symmetric weights for BatchNorm?
#     return BatchUnNormalize(num_features=layer.num_features, dtype=layer.weight.dtype)
