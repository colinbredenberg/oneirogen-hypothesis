from typing import ClassVar

from beyond_backprop.algorithms.reinforce.reinforce import REINFORCE
from beyond_backprop.algorithms.image_classification_test import ImageClassificationAlgorithmTests


class TestFeedbackAlignment(ImageClassificationAlgorithmTests[REINFORCE]):
    """Unit tests for the Feedback Alignment algorithm.

    This runs all the tests from the `AlgorithmTests` class. This should customize how the
    algorithm is created.
    """

    algorithm_type: type[REINFORCE] = REINFORCE
    algorithm_name: str = "reinforce"
    unsupported_datamodule_names: ClassVar[list[str]] = ["imagenet32", "rl", "base", "cartpole", "cifar10", "fashion_mnist", "inaturalist", "moving_mnist", "pendulum"]
    unsupported_network_names: ClassVar[list[str]] = ["resnet18", "resnet34", "fcnet", "custom_lenet", "dagrlmodel", "fcwsmodel", "lenet", "famodel",
                                                      "normmodel", "pcamodel", "resnet18", "resnet34", "rmwsmodel", "simple_vgg", "slmodel", "temporalwsmodel", "wsmodel"]
