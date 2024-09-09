from __future__ import annotations

from logging import getLogger as get_logger
from pathlib import Path
import torch
from lightning import Callback, Trainer
from torch import Tensor
from collections import defaultdict

from beyond_backprop.algorithms.algorithm import Algorithm
from beyond_backprop.utils.types import StepOutputDict
from beyond_backprop.datamodules.dataset_normalizations import cifar10_unnormalization
import matplotlib.pyplot as plt
import numpy as np
logger = get_logger(__name__)

class SimRecord(defaultdict):
    __getattr__= defaultdict.__getitem__
    __setattr__= defaultdict.__setitem__
    __delattr__= defaultdict.__delitem__

class GenerativeSamples(Callback):
    def __init__(self) -> None:
        """
        Callback that plots generative samples from a trained InfGenNetwork after testing
        """
        super().__init__()

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        pl_module.network.reset()
        for tt in range(0, pl_module.time_step_num):
            pl_module.network.gen_forward()
        sample_data = pl_module.network.gen_ts[-1].gen_output.cpu()
        #sample_data = sample_data.view([sample_data.shape[0], 28,28])
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)
        
        fig, axes = plt.subplots(2, 5, sharey = True)
        fig.suptitle('sample generated image')
        idx = 0
        for ii in range(0,2):
            for jj in range(0,5):
                # axes[ii,jj].imshow(cifar10_unnormalization(torch.tensor(sample_data[idx,...].permute(1,2,0))))
                axes[ii,jj].imshow(torch.tensor(sample_data[idx,...].permute(1,2,0)))
                idx = idx + 1
        plt.savefig(str(log_dir / "gen_images"))

        return
    
class MixingSampler(Callback):
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)
        
        fig, axes = plt.subplots(5, 5, sharey = True, sharex = True)
        fig.suptitle('sample mixed image')
        mixing_constant = np.arange(0,1,0.2)
        for jj in range(0,5):
            pl_module.network.reset()
            for tt in range(0, pl_module.time_step_num):
                pl_module.network.mixed_forward(pl_module.x, mixing_constant = 1-mixing_constant[jj])
            for ii in range(0,5):
                sample_data = pl_module.network.ts[0].mixed_output[ii,:].cpu()
                sample_data = sample_data.permute(1,2,0)#.view([28,28])
                # sample_data = cifar10_unnormalization(sample_data)
                sample_data = sample_data
                axes[ii,jj].imshow(sample_data)
                if ii == 0:
                    axes[ii,jj].set_title(r'$\alpha : %1.2f$' %(jj*0.2))
        plt.tight_layout()
        plt.savefig(str(log_dir / "Mixed samples"))

        return