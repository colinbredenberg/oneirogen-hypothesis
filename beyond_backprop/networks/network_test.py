"""Tests for new networks fthat get added to the codebase."""

from typing import ClassVar, Generic, TypeVar

import pytest
import torch
from lightning import seed_everything
from torch import Tensor, nn
from torch.testing._comparison import assert_close

NetworkType = TypeVar("NetworkType", bound=nn.Module)


class NetworkTests(Generic[NetworkType]):
    """Set of unit tests for a a network class."""

    net_type: type[NetworkType]
    device: ClassVar[torch.device] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture()
    def network_kwargs(self):
        """Fixture that returns different parameters to use when creating the network.

        You can customize this for each test, and create other parameterized fixtures for each
        parameter to vary.
        """
        return dict(
            in_channels=3,
            n_classes=10,
            hparams=None,
        )

    @classmethod
    def make_network(cls, *args, **kwargs) -> NetworkType:
        net: NetworkType = cls.net_type(*args, **kwargs)
        assert isinstance(net, nn.Module)
        net = net.to(cls.device)
        return net

    # TODO: Parametrize this with different image sizes from different datamodules!
    @pytest.fixture(params=[32, 28])
    def image_size(self, request: pytest.FixtureRequest):
        return request.param

    @pytest.fixture
    def x(self, network_kwargs: dict) -> Tensor:
        """Fixture that gives network inputs."""
        in_channels = network_kwargs.get("in_channels", 3)
        assert isinstance(in_channels, int)
        return torch.rand([10, in_channels, 32, 32]).to(self.device)

    @pytest.fixture(scope="function")
    def network(self, x: Tensor, network_kwargs: dict) -> NetworkType:
        """Fixture that creates the network used in tests."""
        net = self.make_network(**network_kwargs)
        # perform a forward pass to initialize any lazy weights.
        _ = net(x)
        return net

    @torch.no_grad()
    @pytest.mark.parametrize("seed", [123, 456])
    def test_initialization_is_reproducible(self, seed: int, x: Tensor, network_kwargs: dict):
        seed_everything(seed=seed)
        network_a = self.make_network(**network_kwargs)
        network_a(x)  # Initialize all parameters

        seed_everything(seed=seed)
        network_b = self.make_network(**network_kwargs)
        network_b(x)  # Initialize all parameters

        state_dict_a = network_a.state_dict()
        state_dict_b = network_b.state_dict()
        assert len(state_dict_a) == len(state_dict_b)
        for (key_a, value_a), (key_b, value_b) in zip(state_dict_a.items(), state_dict_b.items()):
            assert key_a == key_b
            assert_close(value_a, value_b)

    @torch.no_grad()
    @pytest.mark.parametrize("seed", [123, 456])
    def test_forward_pass_is_reproducible(self, seed: int, x: Tensor, network_kwargs: dict):
        seed_everything(seed=seed)
        network_a = self.make_network(**network_kwargs)
        y_a = network_a(x)

        seed_everything(seed=seed)
        network_b = self.make_network(**network_kwargs)
        y_b = network_b(x)
        assert_close(y_a, y_b)
