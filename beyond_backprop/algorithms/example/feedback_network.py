"""Module containing the function which creates the feedback network for our example TargetProp
algorithm based on a given forward architecture."""
from __future__ import annotations

import functools

from torch import nn

from beyond_backprop.networks.invertible import Invertible, is_invertible
from beyond_backprop.networks.layers import Reshape, Sequential


@functools.singledispatch
def create_feedback_network(forward_network: Sequential) -> Sequential:
    """Creates the feedback network by "pseudo-inverting" the forward net.

    If the forward network architecture or one of its layers is not supported, a
    `NotImplementedError` is raised.

    TODO: Do we re-add the `init_symmetric_weights` functionality?
    """
    if not isinstance(forward_network, Sequential):
        raise NotImplementedError(
            f"Don't yet know how to create the feedback equivalent of forward networks of type "
            f"{type(forward_network)}. If you think it should, register a handler function for that "
            f"type."
        )
    net = _invert_sequential(forward_network)
    assert isinstance(net, Sequential)
    return net


@functools.singledispatch
def pseudoinvert(layer: nn.Module) -> nn.Module:
    raise NotImplementedError(
        f"Don't yet know how to create the feedback equivalent of forward layer of type "
        f"{type(layer)}. If you think it should, register a handler function for that "
        f"type."
    )


@pseudoinvert.register(nn.Sequential)
def _invert_sequential(module: nn.Sequential) -> Sequential:
    """Returns a Module that can be used to compute or approximate the inverse operation of `self`.

    NOTE: In the case of Sequential, the order of the layers in the returned network
    is reversed compared to the input.
    """
    # NOTE: Inverting a ResNet (which subclasses Sequential) doesn't try to create another ResNet!
    # It just returns a Sequential.
    return Sequential(
        **{name: pseudoinvert(module) for name, module in reversed(list(module._modules.items()))}
    )


pseudoinvert.register(nn.Sequential, _invert_sequential)

# NOTE: All our supported forward networks (LeNet, SimpleVGG) are subclasses of nn.Sequential at
# the moment.


@pseudoinvert.register(nn.Sequential)
def _invert_sequential(module: nn.Sequential) -> Sequential:
    """Returns a Module that can be used to compute or approximate the inverse operation of `self`.

    NOTE: In the case of Sequential, the order of the layers in the returned network
    is reversed compared to the input.
    """
    # NOTE: Inverting a ResNet (which subclasses Sequential) doesn't try to create another ResNet!
    # It just returns a Sequential.
    return Sequential(
        **{name: pseudoinvert(module) for name, module in reversed(list(module._modules.items()))}
    )


@pseudoinvert.register
def pseudoinvert_linear(layer: nn.Linear) -> nn.Linear:
    backward = type(layer)(
        in_features=layer.out_features,
        out_features=layer.in_features,
        bias=layer.bias is not None,
    )
    return backward


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

    # TODO: Test and confirm that these values are correct. I think they are, but I'm not sure.
    backward = nn.ConvTranspose2d(
        in_channels=layer.out_channels,
        out_channels=layer.in_channels,
        kernel_size=(k_h, k_w),
        stride=(s_h, s_w),
        dilation=d_h,
        padding=(p_h, p_w),
        output_padding=(s_h - 1, s_w - 1),
        bias=layer.bias is not None,
        # output_padding=(op_h + 1, op_w + 1),  # Not sure this will always hold
    )
    return backward


@pseudoinvert.register(nn.LazyLinear)
@pseudoinvert.register(nn.LazyConv2d)
def _invert_lazylinear(layer: nn.LazyLinear | nn.LazyConv2d):
    raise RuntimeError(
        "Can't invert Lazy layers. You need to use them in a forward pass first to "
        "instantiate the weights."
    )


@pseudoinvert.register(nn.Identity)
def invert_identity(module: nn.Identity) -> nn.Identity:
    return nn.Identity()


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


@pseudoinvert.register(Reshape)
@pseudoinvert.register(nn.Flatten)
def invert_reshape(module: nn.Flatten | Reshape) -> Reshape:
    assert isinstance(module, Invertible), "mark the net as invertible and do a forward pass first"
    assert module.input_shape and module.output_shape, "Use the net before inverting"
    layer = Reshape(target_shape=module.input_shape)
    layer.output_shape = module.input_shape
    layer.input_shape = module.output_shape
    return layer


@pseudoinvert.register(Reshape)
@pseudoinvert.register(nn.Flatten)
def invert_flatten(module: nn.Flatten | Reshape) -> Reshape:
    assert is_invertible(module), "Use the net before inverting."
    assert isinstance(
        module, Invertible
    ), "Mark the net as invertible and use it in a forward pass first!"
    assert module.input_shape is not None
    layer = Reshape(target_shape=module.input_shape)
    layer.output_shape = module.input_shape
    layer.input_shape = module.output_shape
    return layer


@pseudoinvert.register(nn.BatchNorm2d)
def invert_batchnorm(layer: nn.BatchNorm2d) -> nn.BatchNorm2d:
    raise NotImplementedError("TODO: Double-check that this makes sense.")
    return nn.BatchNorm2d(
        num_features=layer.num_features,
        eps=layer.eps,
        momentum=layer.momentum,
        affine=layer.affine,
        track_running_stats=layer.track_running_stats,
    )
