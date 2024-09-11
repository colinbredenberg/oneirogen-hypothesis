# Codebase for the paper: "The oneirogen hypothesis: modeling the hallucinatory effects of classical psychedelics in terms of replay-dependent plasticity mechanisms"
By Colin Bredenberg, Fabrice Normandin, Blake Richards, and Guillaume Lajoie


This code repository was produced by simplifying and adapting the Mila IDT team's BeyondBackprop library: https://github.com/mila-iqia/BeyondBackprop (still under active development), keeping only code required for reproducing this paper.

Also inspired / adapted from https://www.github.com/ernoult/ScalingDTP as well as https://github.com/ashleve/lightning-hydra-template

## Installation
Step 1: Create a Python 3.9 virtual envioronment.

Step 2: Within that environment, in the command line, run:

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

## Running experiments

To rerun the experiments used to produce the main paper figures, choose an experiment in the ```beyond_backprop/configs/experiment``` folder and from the oneirogen-hypothesis folder run:

```console
python main.py experiment=[experiment_name]
```

experiment_name options: 

Multicompartment model trained on MNIST (main results): ```rm_wake_sleep_ff```
Multicompartment model trained on CIFAR10: ```rm_wake_sleep_ff_cifar```
No learning control: ```rm_wake_sleep_ff_no_learning```
Single compartment model: ```rm_wake_sleep_ff_no_batch_norm```
Noise-based hallucination protocol: ```rm_wake_sleep_ff_noise```
Recurrent network model: ```rm_wake_sleep``` (note: this model is much larger and takes longer to run)



Expected runtime: 2-3 hours. Generating full hallucination videos may take longer, but experiment results will appear in your ```beyond_backprop/logs/``` folder before then.

If you do not have a gpu on your local machine, you may need to change the line:

```console
- override /trainer: default
```
in the config file you would like to run to:

```console
- override /trainer: cpu
```