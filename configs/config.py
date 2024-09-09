from dataclasses import dataclass, field
from logging import getLogger as get_logger
from typing import Optional

from hydra.core.config_store import ConfigStore

from beyond_backprop.algorithms.algorithm import Algorithm
from beyond_backprop.configs.datamodule import DataModuleConfig
from beyond_backprop.networks import Network

logger = get_logger(__name__)


@dataclass
class Config:
    """All the options required for a run. This dataclass acts as a schema for the Hydra configs.

    For more info, see https://hydra.cc/docs/tutorials/structured_config/schema/
    """

    # Configuration for the datamodule (dataset + transforms + dataloader creation).
    datamodule: DataModuleConfig

    # The hyper-parameters of the algorithm to use.
    algorithm: Algorithm.HParams

    # The hyper-parameters of the network to use.
    network: Network.HParams

    # Keyword arguments for the Trainer constructor.
    trainer: dict = field(default_factory=dict)  # type: ignore

    # # Config(s) for the logger(s).
    # logger: Optional[dict] = field(default_factory=dict)  # type: ignore

    # Whether to run in debug mode or not.
    debug: bool = False

    verbose: bool = False

    # Random seed.
    seed: Optional[int] = None

    # Name for the experiment.
    name: str = "default"


Options = Config  # Alias for backward compatibility.

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
