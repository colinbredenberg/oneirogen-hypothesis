from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from .init_symmetric_weights import init_symmetric_weights
from .utils import has_maxpool2d


@torch.no_grad()
def test_can_init_symmetric_weights(forward_network: nn.Module, feedback_network: nn.Module):
    """Can initialize the weights of the pseudoinverse symmetrically with respect to the forward
    network."""
    init_symmetric_weights(forward_network, feedback_network)


@pytest.mark.xfail(reason="TODO: symmetric init doesn't always reduce reconstruction error.")
@torch.no_grad()
def test_init_symmetric_weights_reduces_reconstruction_error(
    forward_network: nn.Module, feedback_network: nn.Module, x_y: tuple[Tensor, Tensor]
):
    """Initializing the weights of the pseudoinverse symmetrically should give a better
    pseudoinverse (not always exact inverse)."""
    x, y = x_y

    max_indices: list[Tensor] = []
    if has_maxpool2d(forward_network):
        output, max_indices = forward_network(x)
    else:
        output = forward_network(x)
    assert isinstance(output, Tensor)

    if has_maxpool2d(forward_network):
        x_hat_before = feedback_network(output, max_indices=max_indices)
    else:
        x_hat_before = feedback_network(output)
    assert isinstance(x_hat_before, Tensor)

    reconstruction_error_random_init = torch.norm(x - x_hat_before)

    assert isinstance(x_hat_before, Tensor)

    init_symmetric_weights(forward_network, feedback_network)

    if has_maxpool2d(forward_network):
        x_hat_after = feedback_network(output, max_indices=max_indices)
    else:
        x_hat_after = feedback_network(output)

    reconstruction_error_symmetric_init = torch.norm(x - x_hat_after)
    assert reconstruction_error_symmetric_init < reconstruction_error_random_init
