from typing import Generic, TypeVar

import pytest
from torch import Tensor

from beyond_backprop.networks.conv_architecture import ConvArchitecture, ConvBlock

NetworkType = TypeVar("NetworkType", bound=ConvArchitecture)


class ConvArchitectureTests(Generic[NetworkType]):
    """Tests for our convolutional architectures (SimpleVGG and LeNet for now)."""

    block_type: type[ConvBlock] = ConvBlock

    @pytest.fixture(params=[True, False])
    def return_maxpool_indices(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture()
    def network_kwargs(self, return_maxpool_indices: bool):
        """Fixture that returns different parameters to use when creating the network.

        You can customize this for each test, and create other parameterized fixtures for each
        parameter to vary.
        """
        return dict(
            in_channels=3,
            n_classes=10,
            hparams=None,
            return_maxpool_indices=return_maxpool_indices,
        )

    def test_network_length_makes_sense(self, network: NetworkType):
        assert len(network) == len(network.blocks) + len(network.fcs)

    def test_forward_gives_indices(
        self, network: NetworkType, x: Tensor, return_maxpool_indices: bool
    ):
        if return_maxpool_indices:
            _, indices = network(x)
            assert isinstance(indices, list)
            assert len(indices) == network._n_blocks
            for idx in indices:
                assert isinstance(idx, Tensor)
        else:
            outputs = network(x)
            assert isinstance(outputs, Tensor)

    def test_blocks_are_correct_type(self, network: NetworkType):
        assert len(network.blocks) == network._n_blocks
        for block in network.blocks:
            assert isinstance(block, self.block_type)
