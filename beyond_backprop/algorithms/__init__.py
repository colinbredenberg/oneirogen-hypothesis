from hydra.core.config_store import ConfigStore

from .algorithm import Algorithm
from .backprop import Backprop
from .wake_sleep.wake_sleep_layered_models import FCWSLayeredModel
from .wake_sleep.wake_sleep_layered_models import LayerwiseDiffusionModel


from .wake_sleep.rm_wake_sleep import RMWakeSleep
from .image_classification import ImageClassificationAlgorithm

# Store the different configuration options for each algorithm.

# NOTE: This works the same way as creating config files for each algorithm under
# `configs/algorithm`. From the command-line, you can select both configs that are yaml files as
# well as structured config (dataclasses).

# If you add a configuration file under `configs/algorithm`, it will also be available as an option
# from the command-line, and be validated against the schema.

_cs = ConfigStore.instance()
# _cs.store(group="algorithm", name="algorithm", node=Algorithm.HParams())
_cs.store(group="algorithm", name="backprop", node=Backprop.HParams())
_cs.store(group="algorithm", name = "rm_wake_sleep", node = RMWakeSleep.HParams())
_cs.store(group= "network", name = "fcwsmodel", node = FCWSLayeredModel.HParams())
_cs.store(group= "network", name = "diffusion_layerwise", node = LayerwiseDiffusionModel.HParams())


__all__ = [
    "Algorithm",
    "Backprop",
    "ImageClassificationAlgorithm",
]
