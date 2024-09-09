"""Common fixtures and utilities for testing the DTP algorithm and its components."""
from __future__ import annotations

import inspect

import pytest
import torch
from torch import Tensor, nn

from beyond_backprop.algorithms.dtp.feedback_networks.pseudoinvert import pseudoinvert
from beyond_backprop.algorithms.dtp.feedback_networks.utils import has_maxunpool2d
from beyond_backprop.networks import SimpleVGG
from beyond_backprop.networks.invertible import (
    set_input_output_shapes_hook,
    set_input_output_shapes_on_forward,
)
from beyond_backprop.networks.lenet import LeNet
from beyond_backprop.networks.network import Network
from beyond_backprop.networks.resnet import ResNet18, ResNet34

num_classes: int = 10


@pytest.fixture(
    params=[
        SimpleVGG,
        LeNet,
        ResNet18,
        ResNet34,
    ]
)
def forward_network(request: pytest.FixtureRequest, x_y: tuple[Tensor, Tensor]):
    # NOTE: This network creation should be seeded properly, since it uses x_y which uses the
    # `seed` fixture.
    x, _ = x_y
    network_type: type[Network] = request.param
    kwargs = {}
    if "return_maxpool_indices" in inspect.signature(network_type.__init__).parameters:
        kwargs["return_maxpool_indices"] = True

    network = network_type(in_channels=x.shape[1], n_classes=num_classes, hparams=None, **kwargs)
    assert isinstance(network, nn.Module)
    network = network.to(x.device)
    # Add the 'check_shapes_hook' to the network to record the shapes of the inputs and outputs,
    # making it easier to create the pseudo-inverse network afterwards.
    set_input_output_shapes_on_forward(network)
    assert set_input_output_shapes_hook in network._forward_hooks.values()
    # Warm-up the network with a forward pass, to instantiate all the layer weights and set all the
    # input-output shapes.
    _ = network(x)
    assert network.input_shape == x.shape[1:]

    return network


@pytest.fixture
def feedback_network(
    forward_network: nn.Module,
    x_y: tuple[Tensor, ...],
) -> nn.Module:
    x, y = x_y
    feedback_network = pseudoinvert(forward_network).to(x.device)
    return feedback_network


@pytest.fixture
def deterministic_feedback_network(feedback_network: nn.Module, request: pytest.FixtureRequest):
    """Fixture that yields the feedback network, but also marks the test with xfail if the network
    uses nn.MaxUnpool2d, since it doesn't currently have a deterministic implementation."""
    mode = torch.get_deterministic_debug_mode()
    torch.set_deterministic_debug_mode("error")
    torch.use_deterministic_algorithms(True)
    if has_maxunpool2d(feedback_network):
        request.node.add_marker(
            pytest.mark.xfail(
                raises=RuntimeError,
                match="max_unpooling2d_forward_out does not have a deterministic implementation",
                reason="max_unpooling2d_forward_out is not deterministic",
            )
        )
        request.node.add_marker(
            pytest.mark.xfail(
                raises=AssertionError,
                match="Values are not sufficiently close",
                reason="max_unpooling2d_forward_out is not deterministic",
            )
        )
        request.node.add_marker(
            pytest.mark.xfail(
                raises=AssertionError,
                match="Tensor-likes are not close!",
                reason="max_unpooling2d_forward_out is not deterministic",
            )
        )
    yield feedback_network
    torch.set_deterministic_debug_mode(mode)
