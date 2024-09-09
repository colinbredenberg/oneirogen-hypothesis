from .inf_gen_network_test import InfGenNetworkTests
from .wake_sleep_layered_models import FCWSLayeredModel


class TestFCWSLayeredModel(InfGenNetworkTests):
    net_type = FCWSLayeredModel
