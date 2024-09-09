"""Module containing the function which creates the forward network for our example TargetProp
algorithm by adapting a base network architecture."""
from __future__ import annotations

import copy
import functools

from torch import nn

from beyond_backprop.networks import LeNet
from beyond_backprop.networks.conv_architecture import ConvArchitecture, ConvBlock
from beyond_backprop.networks.layers import ModuleType, Reshape, Sequential
from beyond_backprop.networks.lenet import LeNetBlock
from beyond_backprop.networks.simple_vgg import SimpleVGG


@functools.singledispatch
def create_forward_network(base_network: nn.Module) -> Sequential:
    """Creates the forward network for the  algorithm from a base network architecture.

    The base network will possibly be modified:
    - nn.MaxPool2d layers are replaced with nn.AvgPool2d;

    If the base network architecture isn't supported, a `NotImplementedError` is raised.
    Same goes for any unsupported layers in the base network.

    NOTE: It would be fairly easy to add support for the ResNet architectures, in the same way as
    was done in DTP. I'm choosing to keep it simple for now, since this is just an example. It also
    illustrates how an algorithm might choose to support only some network architectures.
    """
    raise NotImplementedError(
        f"The given base network of type {type(base_network)} is not yet supported by this "
        "algorithm. If you think it should, register a handler function for that type."
    )


# Covers both SimpleVGG and LeNet, since both are subclasses of ConvArchitecture.
@create_forward_network.register(SimpleVGG)
@create_forward_network.register(LeNet)
@create_forward_network.register(ConvArchitecture)
def _create_forward_net_from_conv_network(base_network: LeNet | SimpleVGG) -> Sequential:
    """Creates the forward network for our algorithm by adapting a base LeNet architecture."""
    # Note: Calling the functions directly, just to be explicit, but in the case of the conv blocks
    # calling `adapt_for_target_prop(conv_block)` would also dispatch to the same function, since
    # it is a registered as a handler for the `ConvBlock` type, which these modules inherit from.
    return Sequential(
        *[_adapt_conv_block(conv_block) for conv_block in base_network.blocks],
        *[_adapt_fc_block(fc_block) for fc_block in base_network.fcs],
    )


@functools.singledispatch
def adapt_for_algorithm(layer: nn.Module) -> nn.Module:
    """Adapts / modifies the given layer so that it is compatible with this algorithm.

    Raises a `NotImplementedError` if the layer isn't supported.
    """
    raise NotImplementedError(
        f"Don't know how to adapt layer of type {type(layer)} for the TargetProp algorithm!\n"
        f"If this layer should be supported, consider registering a handler function for it."
    )


# Covers both SimpleVGGBlock and LeNetBlock, since both are subclasses of ConvBlock.
# @adapt_for_target_prop.register(LeNetBlock)
# @adapt_for_target_prop.register(SimpleVGGBlock)
@adapt_for_algorithm.register(ConvBlock)
def _adapt_conv_block(block: LeNetBlock) -> Sequential:
    return Sequential(
        conv=adapt_for_algorithm(block.conv),
        rho=adapt_for_algorithm(block.rho),
        pool=adapt_for_algorithm(block.pool),
    )


# Note: Not registering this as a handler for nn.Sequential, just to avoid any surprises.
# This handler is called directly in the function above.
def _adapt_fc_block(block: Sequential) -> Sequential:
    return Sequential(
        **{name: adapt_for_algorithm(layer) for name, layer in block.named_children()}
    )


@adapt_for_algorithm.register(Reshape)
@adapt_for_algorithm.register(nn.Flatten)
@adapt_for_algorithm.register(nn.ReLU)
@adapt_for_algorithm.register(nn.ELU)
@adapt_for_algorithm.register(nn.Linear)
@adapt_for_algorithm.register(nn.Conv2d)
def _no_change(layer: ModuleType) -> ModuleType:
    """Just to illustrate, in this case we don't change anything, and just return the same layer.

    NOTE: We return a copy, just to keep things "pure" and to make sure there are no side-effects.
    We don't really need to create a copy, I'm just thinking that it might avoid some bugs later
    down the line.
    """
    return copy.deepcopy(layer)


@adapt_for_algorithm.register(nn.MaxPool2d)
def _replace_maxpool2d_with_avgpool2d(layer: nn.MaxPool2d) -> nn.AvgPool2d:
    """Replaces nn.MaxPool2d layers with an nn.AvgPool2d layer with the same input/output shapes.

    This is helpful, as it makes the resulting forward network easier to "invert" when creating the
    feedback network. When we keep nn.MaxPool2d layers, they become nn.MaxUnpool2d in the feedback
    network, which require the max indices from the forward pass to work.
    Having to pass the forward maxpool indices to the feedback network doesn't seem very
    "bio-plausible" (in my opinion) and also makes the code more complicated.

    This seems more bio-plausible and makes the code cleaner, at the cost of *maybe* degrading
    performance a bit. This remains to be seen.
    """
    return nn.AvgPool2d(
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        padding=layer.padding,
        ceil_mode=layer.ceil_mode,
        # count_include_pad=layer.count_include_pad,
    )
