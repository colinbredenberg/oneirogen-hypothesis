"""Tests for the feedback network creation."""


import pytest
import torch
from pytest_regressions.file_regression import FileRegressionFixture
from torch import Tensor, nn

from beyond_backprop.networks.invertible import set_input_output_shapes_on_forward
from beyond_backprop.networks.network import Network
from beyond_backprop.networks.resnet import ResNet

from .feedback_network import create_feedback_network
from .forward_network import create_forward_network


@pytest.fixture()
def forward_network(base_network: Network, x_y: tuple[Tensor, Tensor]):
    x, _ = x_y
    if isinstance(base_network, (ResNet)):
        pytest.xfail(
            reason="ResNets are purposefully not supported by the example algorithm (for now)."
        )
    forward_network = create_forward_network(base_network)
    forward_network.to(x.device)

    set_input_output_shapes_on_forward(forward_network)
    assert isinstance(forward_network, nn.Module)
    was_training = forward_network.training
    with torch.no_grad():
        forward_network.eval()
        _ = forward_network(x)
        forward_network.train(was_training)
    return forward_network


def test_create_feedback_net(
    forward_network: Network, x_y: tuple[Tensor, Tensor], file_regression: FileRegressionFixture
):
    forward_net = create_feedback_network(forward_network)
    file_regression.check(
        "Forward network:\n"
        + str(forward_network)
        + "\n"
        + "Feedback network:\n"
        + str(forward_net)
    )
