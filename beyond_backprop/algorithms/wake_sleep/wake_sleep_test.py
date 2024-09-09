from typing import ClassVar

from beyond_backprop.algorithms.wake_sleep.wake_sleep import WakeSleep
from beyond_backprop.algorithms.image_classification_test import ImageClassificationAlgorithmTests


class TestFeedbackAlignment(ImageClassificationAlgorithmTests[WakeSleep]):
    """Unit tests for the example algorithm.

    This runs all the tests from the `AlgorithmTests` class. This should customize how the
    algorithm is created.
    """

    algorithm_type: type[WakeSleep] = WakeSleep
    algorithm_name: str = "wake_sleep"
    unsupported_datamodule_names: ClassVar[list[str]] = ["imagenet32", "rl", "base", "cartpole", "cifar10", "fashion_mnist", "inaturalist", "moving_mnist", "pendulum"]
    unsupported_network_names: ClassVar[list[str]] = ["resnet18", "resnet34", "fcnet", "custom_lenet", "dagrlmodel", "fcwsmodel", "lenet", "famodel",
                                                      "resnet18", "resnet34", "rlmodel", "rmwsmodel", "simple_vgg", "temporalwsmodel"]
