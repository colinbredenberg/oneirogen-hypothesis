from beyond_backprop.networks.conv_architecture import ConvBlock
from beyond_backprop.networks.conv_architecture_test import ConvArchitectureTests

from .network_test import NetworkTests
from .simple_vgg import SimpleVGG, SimpleVGGBlock


class TestSimpleVGG(ConvArchitectureTests[SimpleVGG], NetworkTests[SimpleVGG]):
    net_type: type[SimpleVGG] = SimpleVGG
    block_type: type[ConvBlock] = SimpleVGGBlock
