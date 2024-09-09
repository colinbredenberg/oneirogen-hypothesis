from functools import singledispatch

import torch
from torch import nn

from .resnet import BasicBlock, InvertedBasicBlock


@singledispatch
def init_symmetric_weights(forward_layer: nn.Module, feedback_layer: nn.Module) -> None:
    if any(p.requires_grad for p in forward_layer.parameters()):
        raise NotImplementedError(forward_layer, feedback_layer)


@init_symmetric_weights.register
def _weight_b_sym_sequential(forward_layer: nn.Sequential, feedback_layer: nn.Sequential) -> None:
    for f_layer, b_layer in zip(forward_layer, feedback_layer[::-1]):
        init_symmetric_weights(f_layer, b_layer)


@init_symmetric_weights.register(nn.Conv2d)
def _weight_b_sym_conv2d(forward_layer: nn.Conv2d, feedback_layer: nn.ConvTranspose2d) -> None:
    assert forward_layer.weight.shape == feedback_layer.weight.shape, (
        forward_layer.weight.shape,
        feedback_layer.weight.shape,
    )
    with torch.no_grad():
        # NOTE: I guess the transposition isn't needed here?
        feedback_layer.weight.data = forward_layer.weight.data

    if forward_layer.bias is not None:
        assert feedback_layer.bias is not None
        forward_layer.bias.data.zero_()
        feedback_layer.bias.data.zero_()
    else:
        assert feedback_layer.bias is None


@init_symmetric_weights.register(nn.Linear)
def _weight_b_sym_linear(forward_layer: nn.Linear, feedback_layer: nn.Linear) -> None:
    assert forward_layer.in_features == feedback_layer.out_features
    assert forward_layer.out_features == feedback_layer.in_features
    # TODO: Double check that this bias term initialization is ok.
    if forward_layer.bias is None:
        assert feedback_layer.bias is None
    else:
        assert feedback_layer.bias is not None
        forward_layer.bias.data.zero_()
        feedback_layer.bias.data.zero_()

    with torch.no_grad():
        # NOTE: I guess the transposition isn't needed here?
        feedback_layer.weight.data = forward_layer.weight.data.t()


@init_symmetric_weights.register(BasicBlock)
def _init_symmetric_weights_residual(
    forward_layer: BasicBlock, feedback_layer: InvertedBasicBlock
):
    init_symmetric_weights(forward_layer.conv1, feedback_layer.conv1)
    init_symmetric_weights(forward_layer.bn1, feedback_layer.bn1)
    init_symmetric_weights(forward_layer.conv2, feedback_layer.conv2)
    init_symmetric_weights(forward_layer.bn2, feedback_layer.bn2)
    if len(forward_layer.shortcut) > 0:  # Non-identity shortcut
        assert isinstance(forward_layer.shortcut.conv, nn.Conv2d)
        init_symmetric_weights(forward_layer.shortcut.conv, feedback_layer.shortcut.conv)
        if forward_layer.use_batchnorm:
            assert isinstance(forward_layer.shortcut.bn, nn.BatchNorm2d)
            init_symmetric_weights(forward_layer.shortcut.bn, feedback_layer.shortcut.bn)
