"""Tests for the forward network creation."""

import pytest
from pytest_regressions.file_regression import FileRegressionFixture

from beyond_backprop.networks import ResNet
from beyond_backprop.networks.network import Network

from .forward_network import create_forward_network


def test_create_forward_net(base_network: Network, file_regression: FileRegressionFixture):
    if isinstance(base_network, (ResNet)):
        pytest.xfail(
            reason="ResNets are purposefully not supported by the example algorithm (for now)."
        )
    forward_net = create_forward_network(base_network)
    file_regression.check(
        "Base network:\n" + str(base_network) + "\n" + "Forward network:\n" + str(forward_net),
    )
