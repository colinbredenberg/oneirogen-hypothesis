from typing import ClassVar

from beyond_backprop.algorithms.wake_sleep.temporal_wake_sleep import TemporalWakeSleep
from beyond_backprop.algorithms.image_classification_test import ImageClassificationAlgorithmTests


class TestFeedbackAlignment(ImageClassificationAlgorithmTests[TemporalWakeSleep]):
    """Unit tests for the example algorithm.

    This runs all the tests from the `AlgorithmTests` class. This should customize how the
    algorithm is created.
    """

    algorithm_type: type[TemporalWakeSleep] = TemporalWakeSleep
    algorithm_name: str = "temporal_wake_sleep"
    unsupported_datamodule_names: ClassVar[list[str]] = ["imagenet32", "rl", "base", "cartpole", "cifar10", "fashion_mnist", "inaturalist", "moving_mnist", "pendulum"]
    unsupported_network_names: ClassVar[list[str]] = ["resnet18", "resnet34", "fcnet", "custom_lenet", "dagrlmodel", "fcwsmodel", "lenet", "famodel",
                                                      "normmodel", "pcamodel", "resnet18", "resnet34", "rlmodel", "rmwsmodel", "simple_vgg", "slmodel", "wsmodel"]
