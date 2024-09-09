from __future__ import annotations

from logging import getLogger as get_logger
from pathlib import Path
import torch
import torch.nn.functional as F
from lightning import Callback, Trainer
from torch import Tensor
import torchvision
from torchvision.transforms import v2
import cv2
from collections import defaultdict

from beyond_backprop.algorithms.algorithm import Algorithm
from beyond_backprop.utils.types import StepOutputDict
from beyond_backprop.datamodules.dataset_normalizations import cifar10_unnormalization
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
from torch.distributions.multivariate_normal import MultivariateNormal
logger = get_logger(__name__)
basal_color = '#72a6ca'
apical_color = '#e0474c'
fontsize = 5

def generated_image_plot(sample_data, log_dir):
    fig, axes = plt.subplots(2, 5, sharey = True, figsize = (7.5, 3))
    fig.suptitle('sample generated image', fontsize = fontsize)
    idx = 0
    shape = sample_data[idx,...].shape
    if shape[0] == 3 and shape[1] == 32:
        cifar10 = True
    else:
        cifar10 = False
    for ii in range(0,2):
        for jj in range(0,5):
            if cifar10:
                axes[ii,jj].imshow(cifar10_unnormalization(torch.tensor(sample_data[idx,...].permute(1,2,0))))
            else:
                axes[ii,jj].imshow(torch.tensor(sample_data[idx,...].permute(1,2,0)), cmap = 'gray', vmin = -1, vmax = 1)
            axes[ii,jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
            idx = idx + 1
    plt.savefig(str(log_dir / "gen_images.pdf"), format = 'pdf')
    return
    
class GenerativeSamples(Callback):
    def __init__(self) -> None:
        """
        Callback that plots generative samples from a trained InfGenNetwork after testing
        """
        super().__init__()

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        pl_module.network.gen_forward()
        sample_data = pl_module.network.gen_ts[-1].gen_output.cpu()
        #sample_data = sample_data.view([sample_data.shape[0], 28,28])
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)
        
        generated_image_plot(sample_data, log_dir)
        torch.save(torch.tensor(sample_data), str(log_dir / "gen_img_data"))
        return
    
class SimRecord(defaultdict):
    __getattr__= defaultdict.__getitem__
    __setattr__= defaultdict.__setitem__
    __delattr__= defaultdict.__delitem__

def default_factory():
    return []

class BPWakeLossRecord(Callback):
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()
        self.record = SimRecord(default_factory)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Algorithm, outputs, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        self.record.loss.append(pl_module.loss.detach())
        self.record.gen_loss.append(pl_module.pre_grad_gen.detach())
        return

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)
        
        fig, axes = plt.subplots(1, 2, sharey = False)
        fig.suptitle('Loss curves')
        axes[0].plot(torch.tensor(self.record.loss))
        axes[0].set_yscale('symlog')
        axes[0].set_title('cross_ent loss')
        axes[1].plot(torch.tensor(self.record.gen_loss))
        axes[1].set_yscale('symlog')
        axes[1].set_title('gen_loss')
        plt.savefig(str(log_dir / "loss.pdf"), format = 'pdf')
        torch.save(self.record, str(log_dir / "bp_wake_loss_data"))
        return

def inf_gen_loss_plot(record, log_dir):
    fig, axes = plt.subplots(1, 2, sharey = False, figsize = (3,1.5), layout='constrained')
    fig.suptitle('Loss curves', fontsize = fontsize)
    axes[0].plot(torch.tensor(record.inf_loss), color = basal_color)
    axes[0].set_yscale('symlog')
    axes[0].set_title('inf_loss', color = basal_color, fontsize = fontsize)
    axes[1].plot(torch.tensor(record.gen_loss), color = apical_color)
    axes[1].set_yscale('symlog')
    axes[1].set_title('gen_loss', color = apical_color, fontsize = fontsize)
    axes[0].spines.top.set_visible(False)
    axes[0].spines.right.set_visible(False)
    axes[1].spines.top.set_visible(False)
    axes[1].spines.right.set_visible(False)
    # axes[0].locator_params(axis = 'both', tight = True, nbins = 4)
    # axes[1].locator_params(axis = 'both', tight = True, nbins = 4)
    plt.savefig(str(log_dir / "loss.pdf"), format = 'pdf')
    return

class InfGenLossRecord(Callback):
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()
        self.record = SimRecord(default_factory)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Algorithm, outputs, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        self.record.inf_loss.append(pl_module.pre_grad_inf.detach())
        self.record.gen_loss.append(pl_module.pre_grad_gen.detach())
        return

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)

        inf_gen_loss_plot(self.record, log_dir)
        torch.save(self.record, str(log_dir / "inf_gen_loss_data"))
        return

class LossRecord(Callback):
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()
        self.record = SimRecord(default_factory)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: Algorithm, outputs, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        self.record.loss.append(pl_module.loss.detach())
        return

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)
        
        fig, axes = plt.subplots(1, 1, sharey = False)
        fig.suptitle('Loss curve')
        axes.plot(torch.tensor(self.record.loss))
        axes.set_yscale('symlog')
        axes.set_title('loss')

        plt.savefig(str(log_dir / "loss.pdf"), format = 'pdf')
        torch.save(self.record, str(log_dir / "loss_data"))
        return

class WhiteningQuant(Callback):
    "Callback to assess how similar the inference distribution is to the generative distribution"
    def __init__(self) -> None:
        super().__init__()
        self.record = SimRecord(default_factory)
    
    def on_test_batch_end(
        self, trainer: Trainer, pl_module: Algorithm, outputs, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx = 0
        ) -> None:
        pl_module.network.forward(pl_module.x)
        self.record.latent_inf.append(pl_module.network.ts[-1].output)
        pl_module.network.gen_forward()
        self.record.latent_gen.append(pl_module.network.ts[-1].gen_output)

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)

        latent_inf = torch.vstack(self.record.latent_inf).cpu()
        latent_gen = torch.vstack(self.record.latent_gen).cpu()
        fig, axes = plt.subplots(1, 1, sharey = True, sharex = True)
        fig.suptitle('gen vs inf latent distribution comparison')
        
        # sample_data = sample_data
        axes.scatter(latent_inf[:,0], latent_inf[:,1])
        axes.scatter(latent_gen[:,0], latent_gen[:,1])
        plt.tight_layout()
        plt.savefig(str(log_dir / "Latent Distribution Comparison.pdf"), format = 'pdf')
        torch.save(self.record, str(log_dir / "whitening_data"))

def closed_eyes_plot(sample_data, log_dir):
    fig, axes = plt.subplots(5, 5, sharey = True, sharex = True, figsize = (7.5, 7.5))
    fig.suptitle('sample closed-eye images', fontsize = fontsize)
    for ii in range(0,5):
        for jj in range(0,5):
            axes[ii,jj].imshow(sample_data[ii,jj,...], cmap = 'gray', vmin = -1, vmax = 1)
            axes[ii,jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
            if ii == 0:
                axes[ii,jj].set_title(r'$\alpha : %1.2f$' %(jj*0.2), fontsize = fontsize)
    plt.tight_layout()
    plt.savefig(str(log_dir / "Closed-eye samples.pdf"), format = 'pdf')
    return

class ClosedEyes(Callback):
    "Callback for visualizing the effects of closed-eye hallucinations"
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
        
        
        mixing_constant = np.arange(0,1,0.2)
        if torch.cuda.is_available():
            x = torch.zeros(pl_module.x.shape, device = torch.cuda.current_device())
        else:
            x = torch.zeros(pl_module.x.shape)
        
        sample_data = -1 * torch.ones(5,5,*x[0,...].permute(1,2,0).shape) #mnist
        # sample_data = torch.zeros(5,5,*x[0,...].permute(1,2,0).shape) #cifar10
        shape = x[0,...].shape
        if shape[0] == 3 and shape[1] == 32:
            cifar10 = True
        else:
            cifar10 = False
        K = 1
        for ii in range(0,5):
            for jj in range(0,5):
                data = x[[ii],...]
                for kk in range(0,K):
                    pl_module.network.mixed_forward(data, mixing_constant = 1-mixing_constant[jj])
                    data = pl_module.network.ts[0].mixed_output[[ii],:]
                sample_data[ii,jj,...] = data[0,:].cpu().permute(1,2,0) #pl_module.network.ts[0].mixed_output[ii,:].cpu().permute(1,2,0)
                # sample_data[ii,jj,...] = sample_data[ii,jj,...].permute(1,2,0)#.view([28,28])
                if cifar10:
                    sample_data[ii,jj,...] = cifar10_unnormalization(sample_data[ii,jj,...])
                else:
                    sample_data = sample_data

        closed_eyes_plot(sample_data, log_dir)
        torch.save(sample_data, str(log_dir / "closed_eye_sample_data"))
        return

def plasticity_quant_plot(total_apical_plasticity, total_apical_plasticity_sem, total_basal_plasticity, total_basal_plasticity_sem, apical_cossim, basal_cossim, log_dir):
    if len(total_apical_plasticity) == 11:
        mixing_constant = torch.arange(0,1.1,0.1)
    else:
        mixing_constant = torch.arange(0,1,0.2)
    fig, axes = plt.subplots(1,1, figsize = (3,3))
    # axes[0].scatter(mixing_constant, total_apical_plasticity, color = apical_color)
    # axes[1].scatter(mixing_constant, total_basal_plasticity, color = basal_color)
    axes.errorbar(mixing_constant, total_apical_plasticity, yerr = total_apical_plasticity_sem, color = apical_color, ecolor = apical_color)
    axes.errorbar(mixing_constant, (1-mixing_constant) * total_apical_plasticity, yerr = (1-mixing_constant) * total_apical_plasticity_sem, color = 'k', ecolor = 'k')
    axes.set_title('apical')
    # axes[1].set_title('basal')
    plt.legend(['without gating', 'with gating'])
    axes.set_ylim([0, torch.max(total_apical_plasticity)])
    axes.set_xlabel(r'$\alpha$', fontsize = fontsize)
    axes.set_ylabel('total plasticity', fontsize = fontsize)
    # axes[1].set_xlabel(r'$\alpha$', fontsize = fontsize)
    # axes[1].set_ylabel('total plasticity', fontsize = fontsize)
    # axes[0].set_ylim([0, torch.max(total_apical_plasticity)])
    # axes[1].set_ylim([0, torch.max(total_basal_plasticity)])
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    # axes[1].tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    # axes[1].tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    # axes[1].spines.top.set_visible(False)
    # axes[1].spines.right.set_visible(False)

    plt.tight_layout()
    fig.suptitle('Dose dependence of apical plasticity', fontsize = fontsize)
    fig.savefig(str(log_dir / "Plasticity Quant Apical.pdf"), format = 'pdf')

    fig_2, axes_2 = plt.subplots(1,1, figsize = (3,3))
    # axes[0].scatter(mixing_constant, total_apical_plasticity, color = apical_color)
    # axes[1].scatter(mixing_constant, total_basal_plasticity, color = basal_color)
    axes_2.errorbar(mixing_constant, total_basal_plasticity, yerr = total_basal_plasticity_sem, color = basal_color, ecolor = basal_color)
    axes_2.errorbar(mixing_constant, mixing_constant * total_basal_plasticity, yerr = (mixing_constant) * total_basal_plasticity_sem, color = 'k', ecolor = 'k')
    axes_2.set_title('basal')
    # axes[1].set_title('basal')
    plt.legend(['without gating', 'with gating'])
    axes_2.set_ylim([0, torch.max(total_basal_plasticity)])
    axes_2.set_xlabel(r'$\alpha$', fontsize = fontsize)
    axes_2.set_ylabel('total plasticity', fontsize = fontsize)
    # axes[1].set_xlabel(r'$\alpha$', fontsize = fontsize)
    # axes[1].set_ylabel('total plasticity', fontsize = fontsize)
    # axes[0].set_ylim([0, torch.max(total_apical_plasticity)])
    # axes[1].set_ylim([0, torch.max(total_basal_plasticity)])
    axes_2.spines.top.set_visible(False)
    axes_2.spines.right.set_visible(False)
    axes_2.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes_2.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    # axes[1].tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    # axes[1].tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    # axes[1].spines.top.set_visible(False)
    # axes[1].spines.right.set_visible(False)

    plt.tight_layout()
    fig_2.suptitle('Dose dependence of basal plasticity', fontsize = fontsize)
    fig_2.savefig(str(log_dir / "Plasticity Quant Basal.pdf"), format = 'pdf')

    fig_3, axes_3 = plt.subplots(1,2, figsize = (3,1.5))
    axes_3[0].scatter(mixing_constant, apical_cossim, color = apical_color)
    axes_3[1].scatter(mixing_constant, basal_cossim, color = basal_color)
    axes_3[0].set_title('apical')
    axes_3[1].set_title('basal')
    axes_3[0].set_xlabel(r'$\alpha$', fontsize = fontsize)
    axes_3[0].set_ylabel('cosine sim', fontsize = fontsize)
    axes_3[1].set_xlabel(r'$\alpha$', fontsize = fontsize)
    axes_3[1].set_ylabel('cosine sim', fontsize = fontsize)
    axes_3[0].set_ylim([0,1])
    axes_3[1].set_ylim([0,1])
    axes_3[0].spines.top.set_visible(False)
    axes_3[0].spines.right.set_visible(False)
    axes_3[1].spines.top.set_visible(False)
    axes_3[1].spines.right.set_visible(False)
    axes_3[0].tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes_3[0].tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    axes_3[1].tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes_3[1].tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    plt.tight_layout()
    fig_3.suptitle('Dose dependence of plasticity cosine sim', fontsize = fontsize)
    fig_3.savefig(str(log_dir / "Plasticity Quant Cosine Sim.pdf"), format = 'pdf')
    return

@torch.enable_grad()
class PlasticityQuant(Callback):
    "Callback for quantifying the degree of plasticity w/ varying degrees of drug"
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()
        self.record = SimRecord(default_factory)

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)
        gen_opt, inf_opt, _ = pl_module.optimizers()
        with torch.inference_mode(False),torch.set_grad_enabled(True):
            mixing_constant = torch.arange(0,1,0.2)
            mixed_outputs = torch.zeros(*pl_module.network.ts[-1].output.shape, len(mixing_constant))
            total_apical_plasticity = torch.zeros(5)
            total_basal_plasticity = torch.zeros(5)
            for jj in range(0,5):
                pl_module.network.mixed_forward(pl_module.x, mixing_constant = 1-mixing_constant[jj])
                gen_opt.zero_grad()
                total_likelihood_gen_rm = - pl_module.network.gen_log_prob(mixed_output = True)#self.network.diagonal_normal_KL(x, gen = True)#- reward * self.network.gen_log_prob() #
                pre_grad_gen = torch.mean(total_likelihood_gen_rm)
                pre_grad_gen.backward()
                gen_grad_list = []
                for param in pl_module.network.gen_group.parameters():
                    if not(param.grad is None):
                        # gen_grad_list.append(torch.linalg.norm(param.grad))
                        gen_grad_list.append(torch.mean(pl_module.hp.backward_optimizer.lr * torch.abs(param.grad) / torch.abs(param.data)))
                total_apical_plasticity[jj] = torch.mean(torch.tensor(gen_grad_list))
                inf_opt.zero_grad()
                total_likelihood_inf = -pl_module.network.log_prob(mixed_output = True)
                pre_grad_inf = torch.mean(total_likelihood_inf)
                pre_grad_inf.backward()
                inf_grad_list = []
                for param in pl_module.network.inf_group.parameters():
                    if not(param.grad is None):
                        # inf_grad_list.append(torch.linalg.norm(param.grad))
                        inf_grad_list.append(torch.mean(pl_module.hp.forward_optimizer.lr * torch.abs(param.grad) / torch.abs(param.data)))
                total_basal_plasticity[jj] = torch.mean(torch.tensor(inf_grad_list))
        
        plasticity_quant_plot(total_apical_plasticity, total_basal_plasticity, log_dir)
        torch.save(total_apical_plasticity, str(log_dir / "apical_plasticity_data"))
        torch.save(total_basal_plasticity, str(log_dir / "basal_plasticity_data"))

def apical_basal_alignment_plot(record, log_dir):
    plt.figure()
    fig, axes = plt.subplots(1, 5, figsize = (7.5,1.5))
    basal = torch.vstack(record.basal).cpu()
    apical = torch.vstack(record.apical).cpu()
    fig.suptitle('Apical Basal Alignment', fontsize = fontsize)
    for idx in range(0,4):
        axes[idx].scatter(basal[:, idx], apical[:, idx])
        axes[idx].plot([-1, 1], [-1,1])

    basal_standardized = (basal - torch.mean(basal, axis = 0, keepdim = True))/torch.std(basal, axis = 0, keepdim = True)
    apical_standardized = (apical - torch.mean(apical, axis = 0, keepdim = True))/torch.std(apical, axis = 0, keepdim = True)
    corr = basal_standardized.T @ apical_standardized/ basal_standardized.shape[0]
    diag = torch.diag(corr)
    N = len(diag)
    off_diag = corr.flatten()[1:].view(N-1, N+1)[:,:-1].flatten()
    diag_mean = torch.mean(diag)
    off_diag_mean = torch.mean(off_diag)

    diag_std = torch.std(diag)
    off_diag_std = torch.std(off_diag)

    K = 1
    axes[4].imshow(corr[0:K, 0:K], vmin = -1, vmax = 1)
    axes[4].set_title('apical and basal correlations', fontsize = fontsize)
    axes[4].set_ylabel('basal neuron #', fontsize = fontsize)
    axes[4].set_xlabel('apical neuron #', fontsize = fontsize)
    
    fig, axes = plt.subplots(1,1, figsize = (1.5,1.5))
    # axes.bar([0,1], [off_diag_mean, diag_mean], yerr = [off_diag_std, diag_std], color = ['k', 'tab:gray'])
    axes.boxplot([off_diag, diag], tick_labels = ['rand.', 'same'], showfliers = False)
    # axes.plot([0,1], [0,0], 'k')
    # axes.set_xticks([0,1])
    axes.set_yticks([-0.5, 0, 0.5, 1])
    axes.set_ylim([-0.5, 1.2])
    axes.set_yticklabels([-0.5, 0, 0.5, 1], fontsize = fontsize)
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    # axes.set_xticklabels(['rand.', 'same'], fontsize = fontsize, rotation = 90)
    axes.set_ylabel('correlation', fontsize = fontsize)
    plt.tight_layout()
    plt.savefig(str(log_dir / "Apical Basal Alignment.pdf"), format = 'pdf')
    return
    
class ApicalBasalAlignment(Callback):
    "Callback for quantifying the degree of apical-basal alignment for a given network"
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()
        self.record = SimRecord(default_factory)

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: Algorithm, outputs, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx = 0
        ) -> None:
        pl_module.network.forward(pl_module.x)
        self.record.basal.append(pl_module.network.ts[2].output)
        pl_module.network.gen_log_prob()
        self.record.apical.append(pl_module.network.ts[2].predicted_activity_gen[0])
        return

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)

        apical_basal_alignment_plot(self.record, log_dir)
        torch.save(self.record, str(log_dir / "apical_basal_alignment_data"))
        return

def stim_cond_var_plot(mixed_outputs, log_dir):

    #calculate the across-stimulus variance for each neuron
    across_stim_variance = torch.var(mixed_outputs, dim = (0,-1))
    mean_across_stim_variance = torch.mean(across_stim_variance, dim = 0)
    std_across_stim_variance = torch.std(across_stim_variance, dim = 0)

    #calculate the stimulus-conditioned variance for each neuron for each mixing value
    stimulus_conditioned_variance = torch.var(mixed_outputs, dim = -1)
    mean_stimulus_conditioned_variance = torch.mean(stimulus_conditioned_variance, dim = (0,1))
    std_stimulus_conditioned_variance = torch.std(stimulus_conditioned_variance, dim = (0,1))

    #generate plots
    fig, axes = plt.subplots(1,1,figsize = (1.5,1.5))
    plt.bar(torch.arange(0,7), torch.hstack((mean_stimulus_conditioned_variance, mean_across_stim_variance[0])), yerr = torch.hstack((std_stimulus_conditioned_variance, std_across_stim_variance[0])))
    plt.xticks(ticks = torch.arange(0,7), labels = [r"$\alpha$ = 0", r"$\alpha$ = 0.2", r"$\alpha$ = 0.4", r"$\alpha$ = 0.6", r"$\alpha$ = 0.8", r"$\alpha$ = 1", "across stim"], rotation = 90, fontsize = fontsize)
    plt.title("Dose Dependence of Stimulus-Conditioned Variance", fontsize = fontsize)
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    plt.tight_layout()
    plt.savefig(str(log_dir / "Stimulus Conditioned Variance.pdf"), format = 'pdf')
    return

class StimulusConditionedVariance(Callback):
    def __init__(self) -> None:
        "Callback for quantifying the variance of single neurons in response to fixed stimuli as the mixing parameter is increased"
        super().__init__()
        
    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)
        
        repetition_num = 100
        mixing_constant = np.arange(0,1.2,0.2)
        pl_module.network.forward(pl_module.x)
        mixed_outputs = torch.zeros(*pl_module.network.ts[2].output.shape, len(mixing_constant), repetition_num)
        for ii in range(0,repetition_num):
            for jj in range(0,6):
                    pl_module.network.mixed_forward(pl_module.x, mixing_constant = 1-mixing_constant[jj])
                    mixed_outputs[...,jj, ii] = pl_module.network.ts[2].mixed_output.detach()
        stim_cond_var_plot(mixed_outputs, log_dir)
        
        torch.save(mixed_outputs, str(log_dir / "stimulus_conditioned_variance_data"))

def distribution_comparisons_plot(mixed_outputs, log_dir, indicator = ""):
    mixing_constant = np.arange(0,1,0.2)
    fig, axes = plt.subplots(1, 11, sharey = True, sharex = True, figsize = (15,1.5))
    fig_2, axes_2 = plt.subplots(1, 11, sharey = True, sharex = True, figsize = (15,1.5))
    fig_3, axes_3 = plt.subplots(1,1, figsize = (1.5,1.5))
    fig.suptitle('Across stimulus correlation matrices', fontsize = fontsize)
    K = 20
    mixing_constant = np.arange(0,1.1,0.1)
    corr_vec = torch.zeros(len(mixing_constant))
    for jj in range(0,len(mixing_constant)):
        corr_full = torch.corrcoef(mixed_outputs[...,jj].T)
        corr = corr_full[0:K,0:K]
        if jj == 0:
            corr_0 = corr
            corr_0_full = corr_full
        #sample_data = sample_data
        axes[jj].imshow(corr - torch.eye(*corr.shape), vmin = -1, vmax = 1)
        axes[jj].set_title(r'$\alpha : %1.2f$' %(jj*0.1), fontsize = fontsize)
        axes[jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
        axes_2[jj].scatter(corr_0.flatten(), corr.flatten())
        axes_2[jj].set_xlabel(r'$\alpha = 0$ correlation', fontsize = fontsize)
        axes_2[jj].set_ylabel(r'$\alpha = %1.2f$ correlation' %(jj*0.1), fontsize = fontsize)
        corr_vec[jj] = torch.corrcoef(torch.stack([corr_0_full.flatten(), corr_full.flatten()]))[0,1]
        axes_3.scatter(mixing_constant[jj], corr_vec[jj])

    path_1 = indicator + "Correlation matrices.pdf"
    path_2 = indicator + "Correlation matrix scatterplots.pdf"
    fig.savefig(str(log_dir / path_1), format = 'pdf')
    fig_2.savefig(str(log_dir / path_2), format = 'pdf')

    axes_3.set_ylabel('corr sim')
    axes_3.set_title('Dose Dependence of Correlation similarity')
    axes_3.set_ylim([0,1])
    plt.xticks(ticks = mixing_constant, labels = [r"$\alpha$ = 0", r"$\alpha$ = 0.1", r"$\alpha$ = 0.2", r"$\alpha$ = 0.3", r"$\alpha$ = 0.4", r"$\alpha$ = 0.5", r"$\alpha$ = 0.6", r"$\alpha$ = 0.7", r"$\alpha$ = 0.8", r"$\alpha$ = 0.9", r"$\alpha$ = 1.0"], rotation = 90, fontsize = fontsize)
    plt.tight_layout()
    path_3 = indicator + "Correlation similarity metric.pdf"
    fig_3.savefig(str(log_dir / path_3), format = 'pdf')
    return

def repr_sim_plot(mixed_outputs_full, y, log_dir, layer_num):
    num_classes = len(torch.unique(y))
    mixing_constant = np.arange(0,1.1, 0.1)
    corr_sim = torch.zeros(len(mixing_constant))
    if torch.cuda.is_available():
        mixed_outputs_full = mixed_outputs_full.to(torch.cuda.current_device())
    else:
        mixed_outputs_full = mixed_outputs_full
    for jj in range(0, len(mixing_constant)):
        repr_means = torch.stack([torch.mean(mixed_outputs_full[torch.where(y== ii)[0],:, jj], axis = 0) for ii in range(0,num_classes)])
        rdm = 1 - torch.corrcoef(repr_means)
        if jj == 0:
            rdm_0 = rdm
        corr_sim[jj] = torch.corrcoef(torch.vstack([rdm_0.flatten(), rdm.flatten()]))[0,1]

    fig, axes = plt.subplots(1,1, figsize = (1.5,1.5))
    axes.set_ylabel('repr sim')
    axes.set_title('Dose Dependence of Correlation similarity')
    axes.set_ylim([0,1])
    plt.xticks(ticks = mixing_constant, labels = [r"$\alpha$ = 0", r"$\alpha$ = 0.1", r"$\alpha$ = 0.2", r"$\alpha$ = 0.3", r"$\alpha$ = 0.4", r"$\alpha$ = 0.5", r"$\alpha$ = 0.6", r"$\alpha$ = 0.7", r"$\alpha$ = 0.8", r"$\alpha$ = 0.9", r"$\alpha$ = 1.0"], rotation = 90, fontsize = fontsize)
    axes.scatter(mixing_constant, corr_sim)
    filename = "Representational similarity metric " + str(layer_num) + ".pdf"
    fig.savefig(str(log_dir / filename), format = 'pdf')

class DistributionComparisons(Callback):
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()
        self.record = SimRecord(default_factory)

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: Algorithm, outputs, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx = 0
        ) -> None:
        mixing_constant = np.arange(0,1.1,0.1)
        mixed_outputs = torch.zeros(*pl_module.network.ts[2].output.shape, len(mixing_constant))
        mixed_outputs_final = torch.zeros(*pl_module.network.ts[-1].output.shape, len(mixing_constant))
        for jj in range(0,len(mixing_constant)):
                y = pl_module.y
                pl_module.network.mixed_forward(pl_module.x, mixing_constant = 1-mixing_constant[jj])
                mixed_outputs[...,jj] = pl_module.network.ts[2].mixed_output.detach()
                mixed_outputs_final[...,jj] = pl_module.network.ts[-1].mixed_output.detach()
        self.record.mixed_outputs.append(mixed_outputs)
        self.record.mixed_outputs_final.append(mixed_outputs_final)
        self.record.y.append(y)
        return

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)
        
        mixed_outputs = torch.vstack(self.record.mixed_outputs)
        mixed_outputs_final = torch.vstack(self.record.mixed_outputs_final)
        y = torch.hstack(self.record.y)
        distribution_comparisons_plot(mixed_outputs, log_dir)
        repr_sim_plot(mixed_outputs_final, y, log_dir, -1)
        repr_sim_plot(mixed_outputs, y, log_dir, 2)
        torch.save(self.record, str(log_dir / "distribution_comparisons_data"))
        return

def mixed_samples_plot(sample_data, sample_diff, log_dir):
    fig, axes = plt.subplots(5, 5, sharey = True, sharex = True, figsize = (7.5, 7.5))
    fig_2, axes_2 = plt.subplots(5, 5, sharey = True, sharex = True, figsize = (7.5, 7.5))
    fig.suptitle('sample mixed image')
    fig_2.suptitle('sample mixed image')
    mixing_constant = np.arange(0,1,0.2)
    for ii in range(0,5):
        for jj in range(0,5):
            axes[ii,jj].imshow(sample_data[ii,jj,...], cmap = 'gray', vmin = -1, vmax = 1)
            axes_2[ii,jj].imshow(sample_diff[ii,jj,...], cmap = 'gray', vmin = -1, vmax = 1)
            axes[ii,jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
            axes_2[ii,jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
            if ii == 0:
                axes[ii,jj].set_title(r'$\alpha : %1.2f$' %(jj*0.2), fontsize = fontsize)
                axes_2[ii,jj].set_title(r'$\alpha : %1.2f$' %(jj*0.2), fontsize = fontsize)
    fig.tight_layout()
    fig.savefig(str(log_dir / "Mixed samples.pdf"), format = 'pdf')
    fig_2.savefig(str(log_dir / "Mixed samples diff.pdf"), format = 'pdf')
    
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
        
        mixing_constant = np.arange(0,1,0.2)
        sample_data = torch.zeros(5,5,*pl_module.x[0,...].permute(1,2,0).shape)
        sample_diff = torch.zeros(5,5,*pl_module.x[0,...].permute(1,2,0).shape)
        K = 1
        x = pl_module.x
        shape = x[0,...].shape
        if shape[0] == 3 and shape[1] == 32:
            cifar10 = True
        else:
            cifar10 = False
        for ii in range(0,5):
            for jj in range(0,5):
                data = x[[ii],...]
                for kk in range(0,K):
                    pl_module.network.mixed_forward(data, mixing_constant = 1-mixing_constant[jj], pyramid = True)
                    data = pl_module.network.ts[0].mixed_output[[ii],:]
                sample_data[ii,jj,...] = data[0,:].cpu().permute(1,2,0) #pl_module.network.ts[0].mixed_output[ii,:].cpu().permute(1,2,0)
                # sample_data[ii,jj,...] = sample_data[ii,jj,...].permute(1,2,0)#.view([28,28])
                if cifar10:
                    sample_data[ii,jj,...] = cifar10_unnormalization(sample_data[ii,jj,...])
                else:
                    sample_data = sample_data
                sample_diff[ii,jj,...] = sample_data[ii,0,...] - sample_data[ii,jj,...]
        mixed_samples_plot(sample_data, sample_diff, log_dir)
        torch.save(sample_data, str(log_dir / "mixed_samples_data"))

        return

def dynamic_mixed_samples_plot(sample_data, log_dir):
    mixing_constant = np.arange(0,1.1,0.1)
    sample_data[torch.where(sample_data < 0)] = 0
    sample_data[torch.where(sample_data > 1)] = 1
    frame_size = tuple(sample_data.shape[-3:-1])
    for ii in range(0,5):
        for jj in range(0,11):
            file_handle = 'Dynamic mixed samples alpha_' + str(jj) + ' image_' + str(ii) + '.mp4'
            filename = str(log_dir / file_handle)
            output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 60, frame_size)
            T = sample_data.shape[2]
            # data = (cifar10_unnormalization(sample_data[ii,jj,...]) *256).type(torch.uint8).numpy()
            # test_data = np.random.randint(0,256, list(sample_data.shape[2::]), dtype = 'uint8')
            for tt in range(0,T):
                im = ((sample_data[ii,jj,tt,...])*256).type(torch.uint8).numpy() # (cifar10_unnormalization(sample_data[ii,jj,tt,...])*256).type(torch.uint8).numpy()#
                im = im[:,:,[2,1,0]]
                output.write(im)
            output.release()
            cv2.destroyAllWindows()
def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

def figure_to_array(fig):
    fig.canvas.draw()
    fig_array = fig2rgb_array(fig)
    fig_array = fig_array[:,:,[2,1,0]]
    return fig_array

def dynamic_mixed_samples_pyplot(sample_data, log_dir):
    mixing_constant = np.arange(0,1.1,0.1)
    sample_data[torch.where(sample_data < 0)] = 0
    sample_data[torch.where(sample_data > 1)] = 1
    file_handle = 'Dynamic mixed samples image.mp4'
    filename = str(log_dir / file_handle)
    frame_size = (1000, 1000)
    output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 60, frame_size)
    mixing_constant = np.arange(0,1,0.2)
    T = sample_data.shape[2]
    fig, axes = plt.subplots(6, 6, sharey = True, sharex = True, figsize = (10, 10))
    for tt in range(0,T):
        for ii in range(0,6):
            for jj in range(0,6):
                idx_j = jj * 2
                axes[ii,jj].imshow(sample_data[ii,idx_j,tt,...], cmap = 'gray', vmin = -1, vmax = 1)
                axes[ii,jj].tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
                if ii == 0:
                    axes[ii,jj].set_title(r'$\alpha : %1.2f$' %(idx_j*0.2), fontsize = fontsize)
        # fig.tight_layout()
        fig_array = figure_to_array(fig)
        #fig_array = cv2.resize(fig_array, frame_size)
        output.write(fig_array)
    output.release()
    cv2.destroyAllWindows()

def image_flow_corr(static_image, ts_data, of_rad):
    #correlates fluctuations in a hallucinating system with the base 'non-hallucinated' image
    #static image: K x N x N x C
    #optic flow: K x T x N x N x C
    #output: N x N x C
    static_image_centered = static_image - torch.mean(static_image, dim = 0, keepdim = True)
    optic_flow = torch.diff(ts_data, dim = 2)
    optic_flow_centered = optic_flow - torch.mean(optic_flow, dim = (0,1), keepdim = True)
    im_idx = [14, 14]
    static_val = static_image_centered[:,:, im_idx[0], im_idx[1]]
    of_val = optic_flow_centered[:,:,(im_idx[0] - of_rad):(im_idx[1] + of_rad), (im_idx[0] - of_rad):(im_idx[1] + of_rad)]
    cross_cov = torch.mean(static_val.view([static_val.shape[0],1,1,1,1]) * of_val, dim = (0,1,4), keepdim = False)
    return cross_cov

def image_flow_plot(cross_cov, log_dir):
    fig, axes = plt.subplots(1, 11, sharey = True, sharex = True, figsize = (15,1.5))
    fig.suptitle('Spatial correlations', fontsize = fontsize)
    mixing_constant = np.arange(0,1.1,0.1)
    corr_vec = torch.zeros(len(mixing_constant))
    for jj in range(0,cross_cov.shape[0]):
        axes[jj].imshow(cross_cov[jj,...], vmin = -1, vmax = 1)
        axes[jj].set_title(r'$\alpha : %1.2f$' %(jj*0.1), fontsize = fontsize)
        axes[jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
        
    plt.tight_layout()
    fig.savefig(str(log_dir / "spatial_correlation_plot.pdf"), format = 'pdf')


def time_lag_augmentation(data: Tensor, max_lag: int):
    lag_list = []
    for tt in range(1, max_lag + 1):
        lag_list.append(data[(max_lag - tt):(-tt),...])
    
    return torch.cat(lag_list, axis = 2)

def granger_causality(source: Tensor, target: Tensor, max_lag: int = 20):
    "Function that calculates the granger causality from source to target. Tensors are assumed to have dimensions K_1 x T and K_2 x T"
    time_lagged_source = time_lag_augmentation(source, max_lag)
    time_lagged_target = time_lag_augmentation(target, max_lag)

    full_data = torch.cat([time_lagged_source, time_lagged_target], axis = 2)
    N_total = full_data.shape[1]
    N_train = int(N_total * 3/4)
    full_data_train = torch.flatten(full_data[:,0:N_train,...], end_dim = 1)
    full_data_test = torch.flatten(full_data[:,N_train::,...], end_dim = 1)
    reduced_data_train = torch.flatten(time_lagged_target[:,0:N_train,...], end_dim = 1)
    reduced_data_test = torch.flatten(time_lagged_target[:,N_train::,...], end_dim = 1)
    train_target = torch.flatten(target[max_lag::,0:N_train,...], end_dim = 1)
    test_target = torch.flatten(target[max_lag::,N_train::,...], end_dim = 1)

    reg_full = Lasso(alpha = 1e-4).fit(full_data_train, train_target)
    reg_reduced = Lasso(alpha = 1e-4).fit(reduced_data_train, train_target)

    cov_full = torch.cov((test_target - reg_full.predict(full_data_test)).T)
    cov_reduced = torch.cov((test_target - reg_reduced.predict(reduced_data_test)).T)
    return torch.logdet(cov_reduced) - torch.logdet(cov_full)

def granger_causality_plot(gc, indicator, log_dir):
    mixing_constant = torch.arange(0,1.1,0.1)
    fig, axes = plt.subplots(1,1, figsize = (1.5,1.5))
    axes.scatter(mixing_constant, gc, color = apical_color)
    plt.title('granger causality')
    axes.set_xlabel(r'$\alpha$', fontsize = fontsize)
    axes.set_ylabel('F(1 -> 0)', fontsize = fontsize)
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)

    plt.tight_layout()
    path = "granger_causality_quant" + indicator + ".pdf"
    fig.savefig(str(log_dir / path), format = 'pdf')

def wasserstein_2(mean_1, cov_1, mean_2, cov_2):
    mean_1 = mean_1.cpu()
    cov_1 = cov_1.detach().cpu()
    mean_2 = mean_2.cpu()
    cov_2 = cov_2.detach().cpu()
    dist = torch.sum((mean_1 - mean_2)**2) + torch.trace(cov_1 + cov_2 - 2 * sqrtm(cov_1.numpy() @ cov_2.numpy()))
    return dist

def classifier_output_quant(mixed_samples, reference_data, y, network):
    if torch.cuda.is_available():
        mixed_samples = mixed_samples.to(torch.cuda.current_device())
        reference_data = reference_data.to(torch.cuda.current_device())
        y = y.to(torch.cuda.current_device())
    classifier_logits = network.classifier(mixed_samples)
    reference_logits = network.classifier(reference_data)
    class_predictions = torch.argmax(classifier_logits, dim = -1)
    accuracy = torch.sum(class_predictions[:,-1] == y)/len(y)
    variability = torch.mean(torch.var(classifier_logits, axis = 1))

    classifier_mean = torch.mean(classifier_logits.flatten(end_dim = 1).permute(1,0), dim = 1)
    classifier_cov = torch.cov(classifier_logits.flatten(end_dim = 1).permute(1,0))
    classifier_corr = torch.corrcoef(classifier_logits.flatten(end_dim = 1).permute(1,0))
    reference_mean = torch.mean(reference_logits.flatten(end_dim = 1).permute(1,0), dim = 1)
    reference_cov = torch.cov(reference_logits.flatten(end_dim = 1).permute(1,0))
    reference_corr = torch.corrcoef(reference_logits.flatten(end_dim = 1).permute(1,0))

    dist = 1 - torch.corrcoef(torch.stack([classifier_corr.flatten(), reference_corr.flatten()]))[0,1] #wasserstein_2(classifier_mean, classifier_cov, reference_mean, reference_cov)

    return accuracy, variability, dist

def classifier_output_plot(classifier_accuracy, classifier_variability, log_dir):
    mixing_constant = torch.arange(0,1.1,0.1)
    fig, axes = plt.subplots(1,1, figsize = (1.5,1.5))
    axes.scatter(mixing_constant, classifier_accuracy, color = apical_color)
    plt.title('Classifier Accuracy')
    axes.set_xlabel(r'$\alpha$', fontsize = fontsize)
    axes.set_ylabel('Proportion correct', fontsize = fontsize)
    axes.set_ylim([0,1])
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)

    plt.tight_layout()
    fig.savefig(str(log_dir / "classifier accuracy.pdf"), format = 'pdf')

    fig, axes = plt.subplots(1,1, figsize = (1.5,1.5))
    axes.scatter(mixing_constant, classifier_variability, color = apical_color)
    plt.title('Classifier Output Variability')
    axes.set_xlabel(r'$\alpha$', fontsize = fontsize)
    axes.set_ylabel('Variability', fontsize = fontsize)
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)

    plt.tight_layout()
    fig.savefig(str(log_dir / "classifier variability.pdf"), format = 'pdf')

def manifold_distance(test_data, probe_data):
    # test_data = test_data.view([*test_data.shape,1])
    mse = torch.zeros(test_data.shape[0], probe_data.shape[0])
    for ii in range(0,test_data.shape[0]):
        for jj in range(0,probe_data.shape[0]):
            mse[ii,jj] = torch.mean((test_data[ii,...] - probe_data[jj,...])**2, dim = (0,1,2)) #distance_metric(test_data, probe_data)
    vals, idxs = torch.min(mse, dim = 0)
    return torch.mean(vals)

def pointwise_distance(test_data, probe_data):
    errors = torch.mean((test_data - probe_data)**2, dim = (1,2,3))
    return torch.mean(errors)

def manifold_distance_plot(manifold_distance, log_dir):
    mixing_constant = torch.arange(0,1.1,0.1)
    fig, axes = plt.subplots(1,1, figsize = (1.5,1.5))
    axes.scatter(mixing_constant, manifold_distance, color = apical_color)
    plt.title('Manifold distance')
    axes.set_xlabel(r'$\alpha$', fontsize = fontsize)
    axes.set_ylabel('distance', fontsize = fontsize)
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)

    plt.tight_layout()
    fig.savefig(str(log_dir / "manifold distance.pdf"), format = 'pdf')

def pointwise_distance_plot(pointwise_distance, log_dir):
    mixing_constant = torch.arange(0,1.1,0.1)
    fig, axes = plt.subplots(1,1, figsize = (1.5,1.5))
    axes.scatter(mixing_constant, pointwise_distance, color = apical_color)
    plt.title('Pointwise distance')
    axes.set_xlabel(r'$\alpha$', fontsize = fontsize)
    axes.set_ylabel('distance', fontsize = fontsize)
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)

    plt.tight_layout()
    fig.savefig(str(log_dir / "Pointwise distance.pdf"), format = 'pdf')

def dynamic_stim_cond_var(mixed_outputs, reference):
    var = torch.var(mixed_outputs, dim = 0)
    ref_var = torch.var(reference, dim = 0)
    delta_var = torch.mean(var - ref_var)
    sem_var = torch.std(var - ref_var)/np.sqrt(var.shape[0])
    return delta_var, sem_var

# def dynamic_across_stim_var(mixed_outputs, inact_mixed_outputs, reference):
#     ref_var = torch.var(reference, dim = (0,1))
#     var = torch.var(mixed_outputs, dim = (0,1))
#     inact_var = torch.var(inact_mixed_outputs, dim = (0,1))
#     mean_var = torch.mean(var - ref_var)
#     sem_var = torch.std(var - ref_var)/np.sqrt(var.shape[0])

#     mean_inact_var = torch.mean(inact_var - ref_var)
#     sem_inact_var = torch.std(inact_var - ref_var)/np.sqrt(inact_var.shape[0])
#     return mean_var, sem_var, mean_inact_var, sem_inact_var
def dynamic_across_stim_var(mixed_outputs, inact_mixed_outputs):
    var_ratio_eps = 1e-3
    var = torch.var(mixed_outputs[[-1],...], dim = (0,1))
    inact_var = torch.var(inact_mixed_outputs[[-1],...], dim = (0,1))
    mean_var_ratio = torch.mean((inact_var + var_ratio_eps)/(var + var_ratio_eps))
    sem_var_ratio = torch.std((inact_var + var_ratio_eps)/(var + var_ratio_eps))/np.sqrt(var.shape[0])

    return mean_var_ratio, sem_var_ratio

def dynamic_stim_cond_var_plot(mean_stimulus_conditioned_variance, sem_stimulus_conditioned_variance, log_dir, indicator = ""):
    #generate plots
    fig, axes = plt.subplots(1,1,figsize = (1.5,1.5))
    plt.bar(torch.arange(0,12), mean_stimulus_conditioned_variance, yerr = sem_stimulus_conditioned_variance)
    plt.xticks(ticks = torch.arange(0,12), labels = [r"$\alpha$ = 0", r"$\alpha$ = 0.1", r"$\alpha$ = 0.2", r"$\alpha$ = 0.3", r"$\alpha$ = 0.4", r"$\alpha$ = 0.5", r"$\alpha$ = 0.6", r"$\alpha$ = 0.7", r"$\alpha$ = 0.8", r"$\alpha$ = 0.9", r"$\alpha$ = 1", "across stim"], rotation = 90, fontsize = fontsize)
    plt.title("Dose Dependence of Stimulus-Conditioned Variance", fontsize = fontsize)
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    plt.tight_layout()
    path = "Delta Dynamic Stimulus Conditioned Variance" + indicator + ".pdf"
    plt.savefig(str(log_dir / path), format = 'pdf')
    return

def classifier_dist_plot(classifier_dist, log_dir, indicator = ""):
    #generate plots
    fig, axes = plt.subplots(1,1,figsize = (1.5,1.5))
    axes.scatter(torch.arange(0,13), classifier_dist)
    plt.xticks(ticks = torch.arange(0,13), labels = [r"$\alpha$ = 0", r"$\alpha$ = 0.1", r"$\alpha$ = 0.2", r"$\alpha$ = 0.3", r"$\alpha$ = 0.4", r"$\alpha$ = 0.5", r"$\alpha$ = 0.6", r"$\alpha$ = 0.7", r"$\alpha$ = 0.8", r"$\alpha$ = 0.9", r"$\alpha$ = 1", "cov match", "var match"], rotation = 90, fontsize = fontsize)
    plt.title("FID-like Dist. Comp.", fontsize = fontsize)
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    plt.tight_layout()
    path = "FID plot" + indicator + ".pdf"
    plt.savefig(str(log_dir / path), format = 'pdf')
    return

# def dynamic_across_stim_var_plot(mean_delta_variance, std_delta_variance, mean_inact_variance, std_inact_variance, log_dir, indicator = ""):
def dynamic_across_stim_var_plot(mean_var_ratio, std_var_ratio, log_dir, indicator = ""):   
    #generate plots
    fig, axes = plt.subplots(1,1,figsize = (1.5,1.5))
    # plt.errorbar(torch.arange(0,11), mean_delta_variance, yerr = std_delta_variance)
    # plt.errorbar(torch.arange(0,11), mean_inact_variance, yerr = std_inact_variance)
    plt.errorbar(torch.arange(0,11), mean_var_ratio, yerr = std_var_ratio)
    plt.plot(torch.arange(0,11), torch.ones(11), 'k')
    # plt.legend(['no inact', 'inact'], fontsize = fontsize)
    plt.xticks(ticks = torch.arange(0,11), labels = [r"$\alpha$ = 0", r"$\alpha$ = 0.1", r"$\alpha$ = 0.2", r"$\alpha$ = 0.3", r"$\alpha$ = 0.4", r"$\alpha$ = 0.5", r"$\alpha$ = 0.6", r"$\alpha$ = 0.7", r"$\alpha$ = 0.8", r"$\alpha$ = 0.9", r"$\alpha$ = 1"], rotation = 90, fontsize = fontsize)
    plt.title("Dose Dependence of Across-Stim Variance", fontsize = fontsize)
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    axes.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)
    plt.tight_layout()
    path = "Delta Variance" + indicator + ".pdf"
    plt.savefig(str(log_dir / path), format = 'pdf')
    return

def dynamic_image_plot(sample_data, log_dir, indicator = ""):
    fig, axes = plt.subplots(6, 11, sharey = True, figsize = (7.5, 3))
    fig.suptitle('sample generated image', fontsize = fontsize)
    idx = 0
    shape = sample_data[idx,...].shape
    if shape[0] == 3 and shape[1] == 32:
        cifar10 = True
    else:
        cifar10 = False
    for ii in range(0,6):
        for jj in range(0,11):
            if cifar10:
                axes[ii,jj].imshow(cifar10_unnormalization(torch.tensor(sample_data[ii,jj,...].permute(1,2,0))))
            else:
                axes[ii,jj].imshow(torch.tensor(sample_data[ii,jj,...].permute(1,2,0)), cmap = 'gray', vmin = -1, vmax = 1)
            axes[ii,jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
            idx = idx + 1
    path = indicator + "_dynamic_images.pdf"
    plt.savefig(str(log_dir / path), format = 'pdf')
    return

def dynamic_corr_calc(activity):
    flat_activity = activity #activity.reshape([np.prod(activity.shape[0:2]), *activity.shape[2::]])
    return torch.corrcoef(flat_activity.T)

def dynamic_corr_comparisons_plot(corr_list, log_dir, indicator = ""):
    mixing_constant = np.arange(0,1,0.2)
    fig, axes = plt.subplots(1, 11, sharey = True, sharex = True, figsize = (15,1.5))
    fig_2, axes_2 = plt.subplots(1, 11, sharey = True, sharex = True, figsize = (15,1.5))
    fig_3, axes_3 = plt.subplots(1,1, figsize = (1.5,1.5))
    fig.suptitle('Across stimulus correlation matrices', fontsize = fontsize)
    K = 20
    mixing_constant = np.arange(0,1.1,0.1)
    corr_vec = torch.zeros(len(mixing_constant))
    for jj in range(0,len(mixing_constant)):
        corr_full = corr_list[jj,...] #torch.corrcoef(mixed_outputs[...,jj].T)
        corr = corr_full[0:K,0:K]
        if jj == 0:
            corr_0 = corr
            corr_0_full = corr_full
        #sample_data = sample_data
        im = axes[jj].imshow(corr - torch.eye(*corr.shape), vmin = -1, vmax = 1)
        fig.colorbar(im, ax = axes[jj])
        axes[jj].set_title(r'$\alpha : %1.2f$' %(jj*0.1), fontsize = fontsize)
        axes[jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
        axes_2[jj].scatter(corr_0.flatten(), corr.flatten())
        axes_2[jj].set_xlabel(r'$\alpha = 0$ correlation', fontsize = fontsize)
        axes_2[jj].set_ylabel(r'$\alpha = %1.2f$ correlation' %(jj*0.1), fontsize = fontsize)
        corr_vec[jj] = torch.corrcoef(torch.stack([corr_0_full.flatten(), corr_full.flatten()]))[0,1]
        axes_3.scatter(mixing_constant[jj], corr_vec[jj])

    path_1 = indicator + "Correlation matrices.pdf"
    path_2 = indicator + "Correlation matrix scatterplots.pdf"
    fig.savefig(str(log_dir / path_1), format = 'pdf')
    fig_2.savefig(str(log_dir / path_2), format = 'pdf')

    axes_3.set_ylabel('corr sim')
    axes_3.set_title('Dose Dependence of Correlation similarity')
    axes_3.set_ylim([0,1])
    plt.xticks(ticks = mixing_constant, labels = [r"$\alpha$ = 0", r"$\alpha$ = 0.1", r"$\alpha$ = 0.2", r"$\alpha$ = 0.3", r"$\alpha$ = 0.4", r"$\alpha$ = 0.5", r"$\alpha$ = 0.6", r"$\alpha$ = 0.7", r"$\alpha$ = 0.8", r"$\alpha$ = 0.9", r"$\alpha$ = 1.0"], rotation = 90, fontsize = fontsize)
    plt.tight_layout()
    path_3 = indicator + "Correlation similarity metric.pdf"
    fig_3.savefig(str(log_dir / path_3), format = 'pdf')
    return

def cosine_similarity(vec_1, vec_2):
    norm_1 = vec_1.flatten()
    norm_1 = norm_1/torch.linalg.norm(norm_1)
    norm_2 = vec_2.flatten()
    norm_2 = norm_2/torch.linalg.norm(norm_2)

    return torch.dot(norm_1, norm_2)

def fft_analysis(im, log_dir):
    fig, ax = plt.subplots(2, 11, sharey = True, sharex = True, figsize = (25,3))
    ax[0,0].set_ylabel('power')
    ax[1,0].set_ylabel('phase')
    for jj in range(0,11):
        im_fft = torch.fft.fftshift(torch.fft.fft2(im[jj,...]))
        ax[0,jj].set_title(r'$\alpha : %1.2f$' %(jj*0.1), fontsize = fontsize)
        im_1 = ax[0,jj].imshow(torch.log(im_fft.abs()), origin = 'lower')
        im_2 = ax[1,jj].imshow(im_fft.angle(), origin = 'lower')
        ax[0,jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
        ax[1,jj].tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
        fig.colorbar(im_1, ax = ax[0,jj])
        fig.colorbar(im_2, ax = ax[1,jj])
    path = "fft.pdf"
    fig.savefig(str(log_dir / path), format = 'pdf')

def im_corr_analysis(im, log_dir):
    mixing_constant = torch.arange(0,1.1,0.1)
    base_im = im[:,0,...]
    corr_vec = torch.zeros(11)
    for jj in range(0,11):
        base_flat = base_im.flatten(start_dim = 1)
        base_flat = base_flat - torch.mean(base_flat)
        base_flat = base_flat/torch.norm(base_flat, dim = 1, keepdim = True)
        target_flat = im[:,jj,...].flatten(start_dim = 1)
        target_flat = target_flat - torch.mean(target_flat)
        target_flat = target_flat/torch.norm(target_flat, dim = 1, keepdim = True)
        corr_vec[jj] = torch.mean(torch.sum(base_flat * target_flat, dim = 1))

    fig, ax = plt.subplots(1,1, figsize = (1.5, 1.5))
    ax.set_ylabel('corr sim')
    ax.set_title('Dose Dependence of Correlation similarity')
    ax.scatter(mixing_constant, corr_vec)
    ax.set_ylim([0,1])
    ax.set_xlabel(r'$\alpha$', fontsize = fontsize)
    ax.set_ylabel('correlation', fontsize = fontsize)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.tick_params(axis = 'both', which = 'major', labelsize=fontsize)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = fontsize)

    plt.tight_layout()
    fig.savefig(str(log_dir / "image correlation analysis.pdf"), format = 'pdf')

def explained_var_calc(data, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(data.flatten(start_dim = 1))
    return pca.explained_variance_ratio_

def explained_var_plot(explained_var, log_dir):
    fig, ax = plt.subplots(1,1, figsize = (3, 3))
    ax.plot(explained_var[0,:])
    ax.plot(explained_var[5,:])
    ax.plot(explained_var[-1,:])
    ax.set_xlabel('PC #')
    ax.set_ylabel('Proportion explained variance')
    plt.legend((r'$\alpha = 0$', r'$\alpha = 0.5$', r'$\alpha = 1$'))
    plt.tight_layout()
    fig.savefig(str(log_dir / "explained variance ratio analysis.pdf"), format = 'pdf')

class DynamicMixedSampler(Callback):
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()
        self.record = SimRecord(default_factory)

    def on_test_batch_end(
        self, trainer: Trainer, pl_module: Algorithm, outputs, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx = 0
        ) -> None:
        self.record.x.append(pl_module.x)
        self.record.y.append(pl_module.y)
        return
    
    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        with torch.inference_mode(False),torch.set_grad_enabled(True):
            if trainer is not None:
                # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
                # except maybe during some unit tests where the DataModule is used by itself.)
                log_dir = Path(trainer.log_dir or log_dir)
            
            mixing_constant = np.arange(0,1.1,0.1)
            T = 800
            plasticity_eps = 1e-2
            sample_data = torch.zeros(6,11,T, *pl_module.x[0,...].permute(1,2,0).shape)
            
            of_rad = 5
            of_data = torch.zeros(11,of_rad*2, of_rad*2)
            x = pl_module.x
            y = pl_module.y
            timescale = 0.1
            shape = x[0,...].shape
            sample_data_im_analysis = torch.zeros(x.shape[0],11, *pl_module.x[0,...].permute(1,2,0).shape)
            sample_data_closed_eyes = torch.zeros(6,11,*pl_module.x[0,...].permute(1,2,0).shape)
            if shape[0] == 3 and shape[1] == 32:
                cifar10 = True
                inaturalist = False
                mnist = False
            elif shape[0] == 3 and shape[1] == 224:
                inaturalist = True
                cifar10 = False
                mnist = False
            else:
                mnist = True
                cifar10 = False
                inaturalist = False
            gc_data = torch.zeros(11)
            gc_data_reverse = torch.zeros(11)
            classifier_accuracy = torch.zeros(11)
            classifier_variability = torch.zeros(11)
            classifier_dist = torch.zeros(13)
            pointwise = torch.zeros(11)
            manifold = torch.zeros(11)
            test_data = torch.vstack(self.record.x)
            test_labels = torch.vstack(self.record.y)

            # l4_N = np.prod(pl_module.network.ts[3].output.shape[1::]) #np.prod(pl_module.network.ts[4].output.shape[1::])
            l4_N = np.prod(pl_module.network.ts[1].output.shape[1::]) #np.prod(pl_module.network.ts[4].output.shape[1::])

            corr = torch.zeros(11,l4_N, l4_N)
            pc_num = 20
            explained_var = torch.zeros(11, pc_num)

            mean_stimulus_conditioned_variance = torch.zeros(12)
            std_stimulus_conditioned_variance = torch.zeros(12)
            # inact_mean_stimulus_conditioned_variance = torch.zeros(12)
            # inact_std_stimulus_conditioned_variance = torch.zeros(12)
            # mean_delta_variance = torch.zeros(11)
            # sem_delta_variance = torch.zeros(11)
            # mean_inact_variance = torch.zeros(11)
            # sem_inact_variance = torch.zeros(11)'
            mean_var_ratio = torch.zeros(11)
            sem_var_ratio = torch.zeros(11)

            # mean_delta_variance_apical = torch.zeros(11)
            # sem_delta_variance_apical = torch.zeros(11)
            # mean_inact_variance_apical = torch.zeros(11)
            # sem_inact_variance_apical = torch.zeros(11)
            mean_var_ratio_apical = torch.zeros(11)
            sem_var_ratio_apical = torch.zeros(11)
            # inact_mean_across_stim_variance = torch.zeros(11)
            # inact_std_across_stim_variance = torch.zeros(11)
            total_apical_plasticity = torch.zeros(11)
            total_apical_plasticity_sem = torch.zeros(11)
            apical_cosine_sim = torch.zeros(11)
            total_basal_plasticity = torch.zeros(11)
            total_basal_plasticity_sem = torch.zeros(11)
            basal_cosine_sim = torch.zeros(11)
            gen_opt, inf_opt, _ = pl_module.optimizers()
            if mnist:
                multiplier = -1.
            else:
                multiplier = 0.
            if torch.cuda.is_available():
                x_closed_eyes = multiplier*torch.ones(pl_module.x.shape, device = torch.cuda.current_device())
            else:
                x_closed_eyes = multiplier*torch.ones(pl_module.x.shape)
            for jj in range(0,11):
                data = x # torch.zeros(x.shape)#
                # data = pl_module.network.dynamic_mixed_forward(data, T, mixing_constant = 1-mixing_constant[jj], timescale = timescale, idxs = [0,4,-3, -1], mode = pl_module.hallucination_mode)
                # data = pl_module.network.dynamic_mixed_forward(data, T, mixing_constant = 1-mixing_constant[jj], timescale = timescale, idxs = [0,3,-3, -1], mode = pl_module.hallucination_mode)
                data = pl_module.network.dynamic_mixed_forward(data, T, mixing_constant = 1-mixing_constant[jj], timescale = timescale, idxs = [0,1], mode = pl_module.hallucination_mode)


                gen_opt.zero_grad()
                total_likelihood_gen_rm = - pl_module.network.gen_log_prob(mixed_output = True)#self.network.diagonal_normal_KL(x, gen = True)#- reward * self.network.gen_log_prob() #
                pre_grad_gen = torch.mean(total_likelihood_gen_rm)
                pre_grad_gen.backward()
                gen_grad_list = []
                gen_grad_var_list = []
                gen_grad_dim_list = []
                gen_cosine_sim = []
                if jj == 0:
                    baseline_grad_list_gen = []
                ctr = 0
                for param in pl_module.network.gen_group.parameters():
                    if not(param.grad is None):
                        # gen_grad_list.append(torch.linalg.norm(param.grad))
                        if jj == 0:
                            baseline_grad_list_gen.append(param.grad)
                        gen_grad_list.append(torch.mean(pl_module.hp.backward_optimizer.lr * torch.abs(param.grad) / (torch.abs(param.data) + plasticity_eps)))
                        # gen_grad_list.append(torch.mean(torch.clamp(torch.abs(param.grad)/(torch.abs(baseline_grad_list_gen[ctr])+plasticity_eps), max = 10)))
                        gen_grad_dim_list.append(torch.prod(torch.tensor(param.grad.shape)))
                        gen_grad_var_list.append(torch.var(pl_module.hp.backward_optimizer.lr * torch.abs(param.grad) / (torch.abs(param.data) + plasticity_eps)))
                        # gen_grad_var_list.append(torch.var(torch.clamp(torch.abs(param.grad)/(torch.abs(baseline_grad_list_gen[ctr])+plasticity_eps), max = 10)))
                        gen_cosine_sim.append(cosine_similarity(param.grad, baseline_grad_list_gen[ctr]))
                        ctr += 1
                total_apical_plasticity[jj] = torch.mean(torch.tensor(gen_grad_list))
                total_gen_param_num = torch.sum(torch.tensor(gen_grad_dim_list))
                #have to reweight the variances by the number of parameters in each tensor
                total_apical_plasticity_sem[jj] = torch.sum((torch.tensor(gen_grad_dim_list) * torch.tensor(gen_grad_var_list)))/total_gen_param_num / torch.sqrt(total_gen_param_num)
                apical_cosine_sim[jj] = torch.mean(torch.tensor(gen_cosine_sim))
                inf_opt.zero_grad()
                total_likelihood_inf = -pl_module.network.log_prob(mixed_output = True)
                pre_grad_inf = torch.mean(total_likelihood_inf)
                pre_grad_inf.backward()
                inf_grad_list = []
                inf_grad_dim_list = []
                inf_grad_var_list = []
                inf_cosine_sim = []
                if jj == 0:
                    baseline_grad_list_inf = []
                ctr = 0
                for param in pl_module.network.inf_group.parameters():
                    if not(param.grad is None):
                        if jj == 0:
                            baseline_grad_list_inf.append(param.grad)
                        inf_grad_list.append(torch.mean(pl_module.hp.forward_optimizer.lr * torch.abs(param.grad) / (torch.abs(param.data) + plasticity_eps)))
                        # inf_grad_list.append(torch.mean(torch.clamp(torch.abs(param.grad)/(torch.abs(baseline_grad_list_inf[ctr]) + plasticity_eps), max = 10)))
                        inf_grad_dim_list.append(torch.prod(torch.tensor(param.grad.shape)))
                        inf_grad_var_list.append(torch.var(pl_module.hp.forward_optimizer.lr * torch.abs(param.grad) / (torch.abs(param.data) + plasticity_eps)))
                        # inf_grad_var_list.append(torch.var(torch.clamp(torch.abs(param.grad)/(torch.abs(baseline_grad_list_inf[ctr]) + plasticity_eps), max = 10)))
                        inf_cosine_sim.append(cosine_similarity(param.grad, baseline_grad_list_inf[ctr]))
                        ctr +=1
                total_basal_plasticity[jj] = torch.mean(torch.tensor(inf_grad_list))
                total_inf_param_num = torch.sum(torch.tensor(inf_grad_dim_list))
                #have to reweight the variances by the number of parameters in each tensor

                total_basal_plasticity_sem[jj] = torch.sum((torch.tensor(inf_grad_dim_list) * torch.tensor(inf_grad_var_list)))/total_inf_param_num / torch.sqrt(total_inf_param_num)

                #total_basal_plasticity_sem[jj] = torch.std(torch.tensor(inf_grad_list))/np.sqrt(len(inf_grad_list))
                basal_cosine_sim[jj] = torch.mean(torch.tensor(inf_cosine_sim))

                im_data = data[0].cpu().permute(1,0,3,4,2)
                lesion_idx = 0
                # data_inactivation = pl_module.network.dynamic_mixed_forward(x, T, mixing_constant = 1-mixing_constant[jj], timescale = timescale, idxs = [0], lesion_idxs = [9], mode = pl_module.hallucination_mode)
                data_inactivation = pl_module.network.dynamic_mixed_forward(x, T, mixing_constant = 1-mixing_constant[jj], timescale = timescale, idxs = [0], lesion_idxs = [lesion_idx], mode = pl_module.hallucination_mode)

                apical_lesion_idx = len(pl_module.network.gen_ts) - 1
                data_apical_inactivation = pl_module.network.dynamic_mixed_forward(x, T, mixing_constant = 1-mixing_constant[jj], timescale = timescale, idxs = [0], apical_lesion_idxs = [apical_lesion_idx], mode = pl_module.hallucination_mode)
                im_data_inactivation = data_inactivation[0].cpu().permute(1,0,3,4,2)
                im_data_apical_inactivation = data_apical_inactivation[0].cpu().permute(1,0,3,4,2)
                data_closed_eyes = pl_module.network.dynamic_mixed_forward(x_closed_eyes, T, mixing_constant = 1-mixing_constant[jj], timescale = timescale, idxs = [0])
                im_data_closed_eyes = data_closed_eyes[0].cpu().permute(1,0,3,4,2)
                if jj == 0:
                    class_data_ref = data[1].cpu().permute(1,0,2)
                class_data = data[1].cpu().permute(1,0,2)
                corr[jj,...] = dynamic_corr_calc(class_data[:,-1,...])
                explained_var[jj,:] = torch.from_numpy(explained_var_calc(class_data[:,-1,...], pc_num))
                sample_data[:,jj,...] = im_data[0:6,...] #pl_module.network.ts[0].mixed_output[ii,:].cpu().permute(1,2,0)
                sample_data_closed_eyes[:,jj,...] = im_data_closed_eyes[0:6,-1,...]
                sample_data_im_analysis[:,jj,...] = im_data[:,-1,...]
                # sample_data[ii,jj,...] = sample_data[ii,jj,...].permute(1,2,0)#.view([28,28])
                if cifar10:
                    sample_data[:,jj,...] = cifar10_unnormalization(sample_data[:,jj,...])
                    # sample_data_closed_eyes[:,jj,...] = cifar10_unnormalization(sample_data_closed_eyes[:,jj,...])
                elif inaturalist:
                    sample_data = sample_data
                # mean_stimulus_conditioned_variance[jj], std_stimulus_conditioned_variance[jj] = dynamic_stim_cond_var(im_data.permute(1,0,2,3,4))
                # inact_mean_stimulus_conditioned_variance[jj], inact_std_stimulus_conditioned_variance[jj] = dynamic_stim_cond_var(im_data_inactivation.permute(1,0,2,3,4))
                if jj == 0:
                    reference_data = im_data.permute(1,0,2,3,4)
                if jj == 0:
                    mean_stimulus_conditioned_variance[-1], std_stimulus_conditioned_variance[-1] = dynamic_stim_cond_var(im_data.permute(1,0,2,3,4)[0,...], reference_data)
                    # inact_mean_stimulus_conditioned_variance[-1], inact_std_stimulus_conditioned_variance[-1] = dynamic_stim_cond_var(im_data_inactivation.permute(1,0,2,3,4)[0,...])
                mean_stimulus_conditioned_variance[jj], std_stimulus_conditioned_variance[jj] = dynamic_stim_cond_var(im_data.permute(1,0,2,3,4), reference_data)
                # mean_delta_variance[jj], sem_delta_variance[jj], mean_inact_variance[jj], sem_inact_variance[jj] = dynamic_across_stim_var(im_data.permute(1,0,2,3,4), im_data_inactivation.permute(1,0,2,3,4), reference_data)
                # mean_delta_variance_apical[jj], sem_delta_variance_apical[jj], mean_inact_variance_apical[jj], sem_inact_variance_apical[jj] = dynamic_across_stim_var(im_data.permute(1,0,2,3,4), im_data_apical_inactivation.permute(1,0,2,3,4), reference_data)
                mean_var_ratio[jj], sem_var_ratio[jj] = dynamic_across_stim_var(im_data.permute(1,0,2,3,4), im_data_inactivation.permute(1,0,2,3,4))
                mean_var_ratio_apical[jj], sem_var_ratio_apical[jj]= dynamic_across_stim_var(im_data.permute(1,0,2,3,4), im_data_apical_inactivation.permute(1,0,2,3,4))
                # inact_mean_across_stim_variance[jj], inact_std_across_stim_variance[jj] = dynamic_across_stim_var(im_data_inactivation.permute(1,0,2,3,4))
                # gc_data[jj] = granger_causality(data[-1][:,0:200,...], data[-2][:,0:200,...], max_lag = 20)
                # gc_data_reverse[jj] = granger_causality(data[-2], data[-1], max_lag = 20)
                # of_data[jj,:,:] = image_flow_corr(x, im_data, of_rad)
                classifier_accuracy[jj], classifier_variability[jj], classifier_dist[jj] = classifier_output_quant(class_data, class_data_ref, y, pl_module.network)
                # pointwise[jj] = pointwise_distance(x, im_data[:,-1,...].permute(0,3,1,2))
                # manifold[jj] = manifold_distance(test_data, im_data[:,-1,...].permute(0,3,1,2))
            if mnist:
                sample_data = (sample_data + 1)/2 #now mnist data lies between 0 and 1
                sample_data = sample_data.repeat(1,1,1,1,1,3) #convert [28,28,1] shape to [28,28,3] shape
        
        torch.save(sample_data, str(log_dir / "mixed_samples_data"))
        dynamic_stim_cond_var_plot(mean_stimulus_conditioned_variance, std_stimulus_conditioned_variance, log_dir)

        plasticity_quant_plot(total_apical_plasticity, total_apical_plasticity_sem, total_basal_plasticity, total_basal_plasticity_sem, apical_cosine_sim, basal_cosine_sim, log_dir)
        torch.save(total_apical_plasticity, str(log_dir / "apical_plasticity_data"))
        torch.save(total_basal_plasticity, str(log_dir / "basal_plasticity_data"))
        explained_var_plot(explained_var, log_dir)

        # dynamic_across_stim_var_plot(mean_delta_variance, sem_delta_variance, mean_inact_variance, sem_inact_variance, log_dir, indicator = "")
        # dynamic_across_stim_var_plot(mean_delta_variance_apical, sem_delta_variance_apical, mean_inact_variance_apical, sem_inact_variance_apical, log_dir, indicator = " apical inact")
        dynamic_across_stim_var_plot(mean_var_ratio, sem_var_ratio, log_dir, indicator = "")
        dynamic_across_stim_var_plot(mean_var_ratio_apical, sem_var_ratio_apical, log_dir, indicator = " apical inact")
        # dynamic_across_stim_var_plot(inact_mean_across_stim_variance, inact_std_across_stim_variance, log_dir, indicator = "dynamic inact")
        # granger_causality_plot(gc_data, " one to zero", log_dir)
        # granger_causality_plot(gc_data_reverse, " zero to one", log_dir)
        # image_flow_plot(of_data, log_dir)
        classifier_output_plot(classifier_accuracy, classifier_variability, log_dir)
        ref_mean = torch.mean(class_data_ref.flatten(end_dim = 1).permute(1,0), dim = 1)
        ref_cov = torch.cov(class_data_ref.flatten(end_dim = 1).permute(1,0))
        cov_matched_dist = MultivariateNormal(loc = ref_mean, covariance_matrix = ref_cov)
        cov_matched_data = cov_matched_dist.sample(sample_shape = torch.Size([class_data_ref.shape[0], class_data_ref.shape[1]]))
        _, _, classifier_dist[-2] = classifier_output_quant(cov_matched_data.detach(), class_data_ref, y, pl_module.network)
        
        var_matched_dist = MultivariateNormal(loc = ref_mean, covariance_matrix = torch.diag(torch.diag(ref_cov)))
        var_matched_data = var_matched_dist.sample(sample_shape = torch.Size([class_data_ref.shape[0], class_data_ref.shape[1]]))
        _, _, classifier_dist[-1] = classifier_output_quant(var_matched_data, class_data_ref, y, pl_module.network)
        classifier_dist_plot(classifier_dist, log_dir)
        dynamic_image_plot(sample_data[:,:,-1,...].permute(0,1,4,2,3), log_dir, indicator = "mixed")
        dynamic_image_plot(sample_data_closed_eyes.permute(0,1,4,2,3), log_dir, indicator = "closed_eyes")
        dynamic_corr_comparisons_plot(corr, log_dir, indicator = "dynamic_")
        # manifold_distance_plot(manifold, log_dir)
        # pointwise_distance_plot(pointwise, log_dir)
        fft_analysis(sample_data[0,:,-1,:,:,0], log_dir)
        im_corr_analysis(sample_data_im_analysis, log_dir)
        dynamic_mixed_samples_pyplot(sample_data[:,:,0:500,...], log_dir)
        return

class WeightVisualizations(Callback):
    def __init__(self) -> None:
        """
        Callback that records network activation variables for a network during training
        """
        super().__init__()

    def on_test_end(self, trainer: Trainer, pl_module: Algorithm) -> None:
        x = pl_module.x
        shape = x[0,...].shape
        if shape[0] == 3 and shape[1] == 32:
            cifar10 = True
            inaturalist = False
            mnist = False
        elif shape[0] == 3 and shape[1] == 224:
            inaturalist = True
            cifar10 = False
            mnist = False
        else:
            mnist = True
            cifar10 = False
            inaturalist = False
        if trainer is not None:
            # Use the Trainer's log dir if we have a trainer. (NOTE: we should always have one,
            # except maybe during some unit tests where the DataModule is used by itself.)
            log_dir = Path(trainer.log_dir or log_dir)
        
        fig, axes = plt.subplots(5, 5, sharey = True, sharex = True, figsize = (15,15))
        fig.suptitle('1st layer MLP filters')
        mixing_constant = np.arange(0,1,0.2)
        weights = pl_module.network.ts[1].func.mean_func[1].weight.cpu()
        counter = 0
        for ii in range(0,5):
            for jj in range(0,5):
                cifar_dim = [3,32,32]
                mnist_dim = [1,28,28]
                inaturalist_dim = [3,224,224]
                if mnist:
                    im = weights[counter,:].reshape(mnist_dim)
                elif cifar10:
                    im = weights[counter,:].reshape(cifar_dim)
                else:
                    im = weights[counter,:].reshape(inaturalist_dim)
                im = (im - torch.min(im))/(torch.max(im) - torch.min(im))#(im - torch.mean(im))/torch.std(im)
                #im = (im - torch.min(im))/(torch.max(im)-torch.min(im))
                axes[ii,jj].imshow(im.permute((1,2,0)))
                axes[ii,jj].tick_params(left=False,
                    bottom=False,
                    labelleft=False,
                    labelbottom=False)
                counter += 1
        plt.savefig(str(log_dir / "Filter visualization.pdf"), format = 'pdf')

        return