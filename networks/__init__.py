from hydra.core.config_store import ConfigStore


from .lenet import LeNet
from .resnet import ResNet18, ResNet34, ResNet
from .simple_vgg import SimpleVGG
from .fcnet import FcNet
from .network import Network, ImageClassifierNetwork

_cs = ConfigStore.instance()
# _cs.store(group="network", name="network", node=Network.HParams())
_cs.store(group="network", name="simple_vgg", node=SimpleVGG.HParams())
_cs.store(group="network", name="lenet", node=LeNet.HParams())
_cs.store(group="network", name="resnet18", node=ResNet18.HParams())
_cs.store(group="network", name="resnet34", node=ResNet34.HParams())
_cs.store(group="network", name="fcnet", node=FcNet.HParams())

__all__ = [
    "FcNet",
    "ImageClassifierNetwork",
    "LeNet",
    "Network",
    "ResNet",
    "ResNet18",
    "ResNet34",
    "SimpleVGG",
]
