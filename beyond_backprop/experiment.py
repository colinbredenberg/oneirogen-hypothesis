from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Any

from hydra.utils import instantiate
from lightning import Callback, Trainer, seed_everything

from beyond_backprop.algorithms import Algorithm
from beyond_backprop.algorithms.rl_example.rl_datamodule import RlDataModule
from beyond_backprop.configs.config import Config
from beyond_backprop.datamodules.datamodule import DataModule
from beyond_backprop.datamodules.image_classification import ImageClassificationDataModule
from beyond_backprop.networks import Network
from beyond_backprop.networks.fcnet import FcNet, fcnet_for_env
from beyond_backprop.utils.hydra_utils import get_outer_class
from beyond_backprop.utils.types import Dataclass
from beyond_backprop.utils.utils import validate_datamodule
from gym import spaces

logger = get_logger(__name__)


@dataclass
class Experiment:
    """Dataclass containing everything used in an experiment.

    This gets created from the config that are parsed from Hydra. Can be used to run the experiment
    by calling `run(experiment)`. Could also be serialized to a file or saved to disk, which might
    come in handy with `submitit` later on.
    """

    algorithm: Algorithm
    network: Network
    datamodule: DataModule
    trainer: Trainer


def setup_experiment(experiment_config: Config) -> Experiment:
    """Do all the postprocessing necessary (e.g., create the network, Algorithm, datamodule,
    callbacks, Trainer, etc) to go from the options that come from Hydra, into all required
    components for the experiment, which is stored as a namedtuple-like class called `Experiment`.

    NOTE: This also has the effect of seeding the random number generators, so the weights that are
    constructed are always deterministic.
    """

    root_logger = logging.getLogger()
    if experiment_config.debug:
        root_logger.setLevel(logging.INFO)
    elif experiment_config.verbose:
        root_logger.setLevel(logging.DEBUG)

    if experiment_config.seed is not None:
        seed = experiment_config.seed
        print(f"seed manually set to {experiment_config.seed}")
    else:
        seed = random.randint(0, int(1e5))
        print(f"Randomly selected seed: {seed}")
    seed_everything(seed=seed, workers=True)

    # NOTE: Need to do a bit of sneaky type tricks to convince the outside world that these
    # fields have the right type.

    # instantiate all the callbacks
    callbacks: dict[str, Callback] = instantiate(experiment_config.trainer.pop("callbacks"))
    # Create the loggers, if any.
    loggers: dict[str, Any] | None = instantiate(experiment_config.trainer.pop("logger", {}))
    # Create the Trainer.
    assert isinstance(experiment_config.trainer, dict)
    if experiment_config.debug:
        logger.info("Setting the max_epochs to 1, since the 'debug' flag was passed.")
        experiment_config.trainer["max_epochs"] = 1
    if "_target_" not in experiment_config.trainer:
        experiment_config.trainer["_target_"] = Trainer

    trainer = instantiate(
        experiment_config.trainer,
        callbacks=list(callbacks.values()),
        logger=list(loggers.values()) if loggers else None,
    )
    assert isinstance(trainer, Trainer)
    trainer = trainer

    # Create the datamodule:
    datamodule_config: Dataclass = experiment_config.datamodule
    datamodule_overrides = {}
    if hasattr(experiment_config.algorithm, "batch_size"):
        # The algorithm has the batch size as a hyper-parameter.
        algo_batch_size = getattr(experiment_config.algorithm, "batch_size")
        assert isinstance(algo_batch_size, int)
        logger.info(
            f"Overwriting `batch_size` from datamodule config with the value on the Algorithm "
            f"hyper-parameters: {algo_batch_size}"
        )
        datamodule_overrides["batch_size"] = algo_batch_size
    datamodule = instantiate(datamodule_config, **datamodule_overrides)
    datamodule = validate_datamodule(datamodule)

    # Create the network
    network_hparams: Network.HParams = experiment_config.network
    network = instantiate_network(network_hparams=network_hparams, datamodule=datamodule)

    # Create the algorithm
    algo_hparams: Algorithm.HParams = experiment_config.algorithm
    algorithm_type: type[Algorithm] = get_outer_class(type(algo_hparams))
    assert isinstance(
        algo_hparams, algorithm_type.HParams  # type: ignore
    ), "HParams type should match model type"

    algorithm = algorithm_type(
        datamodule=datamodule,
        network=network,
        hp=algo_hparams,
    )
    return Experiment(
        trainer=trainer,
        algorithm=algorithm,
        network=network,
        datamodule=datamodule,
    )


def instantiate_network(network_hparams: Network.HParams, datamodule: DataModule) -> Network:
    network_type: type[Network] = get_outer_class(type(network_hparams))
    assert isinstance(network_hparams, network_type.HParams), "HParams type should match net type"
    if isinstance(datamodule, ImageClassificationDataModule):
        # if issubclass(network_type, ImageClassifierNetwork):
        network = network_type(
            in_channels=datamodule.dims[0],
            n_classes=datamodule.num_classes,  # type: ignore
            hparams=network_hparams,
        )

    elif isinstance(datamodule, RlDataModule):
        # TODO: Make this more general: Reinforce should be able to use other architectures on SL
        # problems also.
        if issubclass(network_type, FcNet):
            assert isinstance(network_hparams, FcNet.HParams)
            network = fcnet_for_env(
                observation_space=datamodule.env.observation_space,
                action_space=datamodule.env.action_space,
                hparams=network_hparams,
            )
        else:
            # TODO: These networks assume that the input are images. For now we tried CartPole with
            # Reinforce, but we could potentially try other Gym envs with Pixel observations.
            assert isinstance(datamodule.env.observation_space, spaces.Box)
            assert len(datamodule.env.observation_space.shape) == 3
            n_channels = datamodule.env.observation_space.shape[0]
            assert isinstance(datamodule.env.action_space, spaces.Discrete)
            n_classes = datamodule.env.action_space.n
            # TODO: Make this a bit more generic perhaps?
            network = network_type(
                in_channels=n_channels,
                n_classes=n_classes,
                hparams=network_hparams,
            )
    else:
        raise NotImplementedError(datamodule, network_hparams)
    assert network.hparams is network_hparams
    return network
