# WIP: Codebase for biologically-inspired learning algorithms (alternatives to backprop)

Inspired / adapted from https://www.github.com/ernoult/ScalingDTP as well as https://github.com/ashleve/lightning-hydra-template

## Installation

```console
pip install -e .
```

## Codebase structure

The main logic of our method is in [beyond_backprop/algorithms/algorithm.py](beyond_backprop/algorithms/algorithm.py)

This Algorithm class is based on the LightningModule class of [PyTorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning).

Here is how the codebase is roughly structured:

```
├── main.py                # training script
└── beyond_backprop
    ├── algorithms     # Learning algorithm implementations
    ├── configs        # configuration files and classes for Hydra
    ├── datamodules    # Datasets and dataloaders
    ├── networks       # Network definitions
    └── utils          # Utility functions
```

## Running the code

This codebase uses [Hydra](https://hydra.cc/) for configuration management and for the command-line interface.
You can see all the command-line groups available with:

```console
python main.py --help
```

From there, you can see all the available options for each group.
The three most important groups are "algorithm", "datamodule", and "network".

You can select a configuration from each group, as well as overwrite any of the options in the
configuration.

```console
python main.py algorithm=backprop datamodule=mnist trainer.max_epochs=10
```

## Running the tests

Install the dev and test dependencies with:

```console
pip install -e .[dev,test]
```

```console
pytest
```

Note: running the unit tests currently takes a little while, and possibly assumes you have a GPU.

You can additionally use pre-commit for auto-formatting and linting if you want:

```console
pre-commit install
```
