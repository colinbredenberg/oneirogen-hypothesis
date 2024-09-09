from beyond_backprop.networks.network_test import NetworkTests
from torch import Tensor
from .inf_gen_network import InfGenNetwork


class InfGenNetworkTests(NetworkTests):
    net_type: type[InfGenNetwork]

    def test_forward(self, network: InfGenNetwork, x: Tensor):
        ...

    def test_specific_to_inf_gen_networks(self):
        ...
