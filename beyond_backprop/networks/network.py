from __future__ import annotations

from dataclasses import dataclass
from logging import getLogger as get_logger
from typing import Protocol

logger = get_logger(__name__)

# IDEA: Add a "min_image_dim" attribute to the HParams class, and use it to check that the
# datamodule has big enough images to be used with the network.


class Network(Protocol):
    """Protocol that describes what we expect to find as attributes and methods on the network that
    is passed to the Algorithms.

    Networks don't necessarily need to inherit from this, they just need to match the attributes
    and methods defined here.

    - They need to have a `hparams` attribute
    """

    @dataclass
    class HParams:
        """Dataclass containing the parameters that control the architecture of the network."""

    hparams: HParams


class ImageClassifierNetwork(Network, Protocol):
    """Protocol that describes what we expect to find as attributes and methods on the network that
    is passed to the Algorithms.

    Networks don't necessarily need to inherit from this, they just need to match the attributes
    and methods defined here.

    - They need to have a `hparams` attribute
    - Their constructor needs to accept `in_channels`, `n_classes` and `hparams` as arguments.
    """

    @dataclass
    class HParams(Network.HParams):
        """Dataclass containing the parameters that control the architecture of the network."""

    hparams: HParams

    def __init__(self, in_channels: int, n_classes: int, hparams: HParams | None = None):
        ...
