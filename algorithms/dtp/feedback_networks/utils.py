from torch import Tensor, nn

from beyond_backprop.algorithms.dtp.feedback_networks.conv_feedback_network import (
    ConvFeedbackNetwork,
)


def has_maxpool2d(network: nn.Module) -> bool:
    for module in network.modules():
        if isinstance(module, nn.MaxPool2d):
            return True
    return False


def has_maxunpool2d(network: nn.Module) -> bool:
    for module in network.modules():
        if isinstance(module, nn.MaxUnpool2d):
            return True
    return False


def forward_pass(forward_network: nn.Module, input: Tensor) -> tuple[Tensor, list[Tensor]]:
    max_indices: list[Tensor] = []
    if has_maxpool2d(forward_network):
        output, max_indices = forward_network(input)
    else:
        output = forward_network(input)
    return output, max_indices


def feedback_pass(
    feedback_network: nn.Module, input: Tensor, forward_max_indices: list[Tensor]
) -> Tensor:
    if has_maxunpool2d(feedback_network):
        assert isinstance(feedback_network, ConvFeedbackNetwork)
        return feedback_network(input, forward_maxpool_indices=forward_max_indices)
    return feedback_network(input)
