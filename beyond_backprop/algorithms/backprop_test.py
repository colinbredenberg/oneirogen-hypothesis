from typing import ClassVar

from beyond_backprop.algorithms.image_classification_test import ImageClassificationAlgorithmTests
from .backprop import Backprop


class TestBackprop(ImageClassificationAlgorithmTests[Backprop]):
    algorithm_type = Backprop
    algorithm_name: str = "backprop"
    unsupported_datamodule_names: ClassVar[list[str]] = ["rl"]
