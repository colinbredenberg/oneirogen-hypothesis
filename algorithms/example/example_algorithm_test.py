from typing import ClassVar

from beyond_backprop.algorithms.example.example_algorithm import ExampleAlgorithm
from beyond_backprop.algorithms.image_classification_test import ImageClassificationAlgorithmTests


class TestExampleAlgorithm(ImageClassificationAlgorithmTests[ExampleAlgorithm]):
    """Unit tests for the example algorithm.

    This runs all the tests from the `AlgorithmTests` class. This should customize how the
    algorithm is created.
    """

    algorithm_type: type[ExampleAlgorithm] = ExampleAlgorithm
    algorithm_name: str = "example"
    unsupported_datamodule_names: ClassVar[list[str]] = ["imagenet32", "rl"]
    unsupported_network_names: ClassVar[list[str]] = ["resnet18", "resnet34", "fcnet"]
