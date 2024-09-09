from typing import ClassVar

from beyond_backprop.algorithms.rl_example.dag_reinforce import DAGREINFORCE
from beyond_backprop.algorithms.image_classification_test import ImageClassificationAlgorithmTests


class TestFeedbackAlignment(ImageClassificationAlgorithmTests[DAGREINFORCE]):
    """Unit tests for the example algorithm.

    This runs all the tests from the `AlgorithmTests` class. This should customize how the
    algorithm is created.
    """

    algorithm_type: type[DAGREINFORCE] = DAGREINFORCE
    algorithm_name: str = "dag_reinforce"
    unsupported_datamodule_names: ClassVar[list[str]] = ["imagenet32", "rl", "base", "cartpole", "cifar10", "fashion_mnist", "inaturalist", "moving_mnist", "pendulum"]
    unsupported_network_names: ClassVar[list[str]] = ["resnet18", "resnet34", "fcnet", "custom_lenet", "fcwsmodel", "lenet", "famodel",
                                                      "normmodel", "pcamodel" "resnet18", "resnet34", "rlmodel", "rmwsmodel", "simple_vgg", "slmodel", "temporalwsmodel", "wsmodel"]
