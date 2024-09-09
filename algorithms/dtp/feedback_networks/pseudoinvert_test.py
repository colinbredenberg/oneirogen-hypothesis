"""Tests for new networks fthat get added to the codebase."""
from __future__ import annotations

from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.file_regression import FileRegressionFixture
from torch import Tensor, nn
from torch.testing._comparison import assert_close

from .utils import feedback_pass, forward_pass


def test_architecture(
    forward_network: nn.Module,
    feedback_network: nn.Module,
    file_regression: FileRegressionFixture,
):
    # Print out the network architecture for the forward and feedback networks to a file.
    # This file is then checked against the expected output in the test to make sure we don't
    # unintentionally change the architecture later.
    file_regression.check(
        "forward net:\n" + str(forward_network) + "\nfeedback net:\n" + str(feedback_network)
    )


def test_forward_pass_is_reproducible(
    x_y: tuple[Tensor, Tensor],
    forward_network: nn.Module,
):
    x, y = x_y

    output_b = forward_pass(forward_network, x)
    output_a = forward_pass(forward_network, x)
    assert_close(output_a, output_b)


def test_feedback_pass_is_reproducible(
    x_y: tuple[Tensor, Tensor],
    forward_network: nn.Module,
    deterministic_feedback_network: nn.Module,
):
    x, y = x_y

    output, max_indices = forward_pass(forward_network, x)
    deterministic_feedback_network.eval()

    x_hat_a = feedback_pass(deterministic_feedback_network, output, max_indices)
    assert isinstance(x_hat_a, Tensor)

    x_hat_b = feedback_pass(deterministic_feedback_network, output, max_indices)
    assert isinstance(x_hat_b, Tensor)

    assert_close(x_hat_a, x_hat_b)


def test_forward_pass_doesnt_change_over_time(
    x_y: tuple[Tensor, Tensor],
    forward_network: nn.Module,
    ndarrays_regression: DataRegressionFixture,
):
    """Test that the forward pass computation doesn't change over time.

    This is a regression test: It checks that we don't accidentally change the forward pass of the
    network during development. If you change the network architecture, you should update the
    file in the test data directory using the --force-regen option with that specific test, or
    --regen-all to update the test data for all tests.
    """
    x, y = x_y

    output, max_indices = forward_pass(forward_network, x)
    ndarrays_regression.check(
        {
            "x": x.cpu().numpy(),
            "y": y.cpu().numpy(),
            "output": output.detach().cpu().numpy(),
            **{
                f"max_indices_{i}": index.detach().cpu().numpy()
                for i, index in enumerate(max_indices)
            },
        }
    )


def test_feedback_pass_doesnt_change_over_time(
    x_y: tuple[Tensor, Tensor],
    forward_network: nn.Module,
    deterministic_feedback_network: nn.Module,
    ndarrays_regression: DataRegressionFixture,
):
    """Regression test: Check that the forward pass of the feedback network doesn't change over
    time."""
    x, y = x_y
    output, max_indices = forward_pass(forward_network, x)
    x_hat = feedback_pass(deterministic_feedback_network, output, max_indices)
    ndarrays_regression.check(
        {
            "x": x.cpu().numpy(),
            "y": y.cpu().numpy(),
            "x_hat": x_hat.detach().cpu().numpy(),
            "output": output.detach().cpu().numpy(),
            **{
                f"max_indices_{i}": index.detach().cpu().numpy()
                for i, index in enumerate(max_indices)
            },
        }
    )
