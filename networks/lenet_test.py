from beyond_backprop.networks.conv_architecture import ConvBlock
from beyond_backprop.networks.conv_architecture_test import ConvArchitectureTests

from .lenet import LeNet, LeNetBlock
from .network_test import NetworkTests


class TestLeNet(ConvArchitectureTests[LeNet], NetworkTests[LeNet]):
    net_type: type[LeNet] = LeNet
    block_type: type[ConvBlock] = LeNetBlock
