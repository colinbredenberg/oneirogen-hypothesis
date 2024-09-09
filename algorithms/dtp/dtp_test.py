import dataclasses
from pathlib import Path
from typing import ClassVar

import pytest
from torch import Tensor, nn
from beyond_backprop.algorithms.image_classification_test import ImageClassificationAlgorithmTests

from beyond_backprop.configs.config import Config
from beyond_backprop.datamodules.vision_datamodule import VisionDataModule
from beyond_backprop.networks.resnet import ResNet

from .dtp import DTP

CONFIG_DIR = str(Path(__file__).parent.parent.parent / "configs")


class TestDTP(ImageClassificationAlgorithmTests[DTP]):
    algorithm_name: ClassVar[str] = "dtp"

    # TODO: Support the simple FCNET with DTP (should be pretty straight-forward to do).
    unsupported_network_names: ClassVar[list[str]] = ["fcnet"]
    unsupported_datamodule_names: ClassVar[list[str]] = ["rl"]

    # TODO: The tests take quite a long time to run because the number of feedback training
    # iterations per layer is quite large!

    @pytest.fixture
    def hp(self, hydra_options: Config) -> DTP.HParams:
        """Fixture that returns the algorithm hyperparameters to use during tests."""
        algo_hp = hydra_options.algorithm
        assert isinstance(algo_hp, DTP.HParams)
        iterations = algo_hp.feedback_training_iterations
        return dataclasses.replace(
            algo_hp,
            feedback_training_iterations=[1 for _ in iterations],
            # TODO: The `test_gradient_step_lowers_loss` test below is flaky. The loss sometimes
            # increases after just one step. Perhaps decreasing the learning rate might help?
            # b_optim=dataclasses.replace(
            #     algo_hp.b_optim, lr=[lr_i / 10 for lr_i in algo_hp.b_optim.lr]
            # ),
            f_optim=dataclasses.replace(algo_hp.f_optim, lr=algo_hp.f_optim.lr / 10),
        )

    @pytest.mark.xfail(
        reason="TODO: This test is flaky, the DTP loss sometimes increases after just one step."
    )
    def test_training_step_lowers_loss(
        self,
        algorithm: DTP,
        training_batch: tuple[Tensor, Tensor],
        accelerator: str,
        request: pytest.FixtureRequest,
    ):
        if isinstance(algorithm.forward_network, ResNet):
            request.node.add_marker(
                pytest.mark.xfail(reason="ResNets don't seem to be working well in this test.")
            )
        return super().test_training_step_lowers_loss(
            algorithm=algorithm, training_batch=training_batch, accelerator=accelerator
        )

    def test_model_network_overrides_fixes_mismatch_in_number_of_values(
        self,
        datamodule: VisionDataModule,
        hydra_options: Config,
    ) -> None:
        """Checks that for every possible datamodule x network combination, the number of
        iterations per layer, the noise per layer, and the learning rate per layer are all.

        of the right length, namely `len(algorithm.feedback_network) - 1`.
        """
        options = hydra_options
        assert isinstance(options, Config)
        assert isinstance(options.algorithm, DTP.HParams)

        n_values_in_algo_config = len(options.algorithm.feedback_training_iterations)
        assert n_values_in_algo_config == len(options.algorithm.b_optim.lr)
        assert n_values_in_algo_config == len(options.algorithm.noise)

        from beyond_backprop.experiment import instantiate_network

        base_network = instantiate_network(network_hparams=options.network, datamodule=datamodule)
        assert isinstance(base_network, nn.Sequential)
        # NOTE: -1 since the first feedback layer (closest to the input x) isn't trained.
        n_feedback_layers_to_train = len(base_network) - 1

        assert n_values_in_algo_config == n_feedback_layers_to_train
