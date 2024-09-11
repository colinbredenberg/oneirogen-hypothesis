"""
Classes for defining individual layers

InfGenProbabilityLayer is the basic layer unit for our Wake-Sleep networks

IGFuncLayerWrapper is a convenient wrapper for turning a top-down layer and a bottom-up layer into a single InfGenProbabilityLayer

The remaining layers are used as inputs to IGFuncLayerWrapper
"""

from __future__ import annotations
from typing import Callable
import torch
from torch import nn, Tensor

import numpy as np

class InfGenProbabilityLayer(nn.Module):
    """
    Layer that it defines a bottom-up inference distribution and a top-down generative distribution,
    with separate distribution parameters, which are sampled through either forward() for gen_forward() respectively.

    Information about both outputs and both distribution parameters are stored separately for gradient computation.

    Attributes
    ----------
    x: the layer's inputs
    dist: the network's inference distribution
    gen_dist: the network's generative distribution
    differentiable: Boolean indicating whether or not the network should register its sampling with autograd

    dist_params: inputs necessary for sampling from self.dist
    gen_dist_params: inputs necessary for sampling from self.gen_dist

    output: the sampled inference output of the probability layer
    gen_output: the sampled generative output of the probability layer

    Methods
    ----------
    forward:
        the layer processes its inputs, sets dist_params, and sets output
    sample:
        the layer produces a sample based on its distribution parameters
    calc_dist_params:
        calculates the parameters for the layer's distribution (subclasses will use nn.Modules to do this)
    log_prob:
        calculates the log probability of the layer's output sample wrt its own probability distribution
        if inputs are included, the method will automatically use calc_dist_params to calculate the log probability
    """
    def __init__(self, 
                 dists: list[Callable[[Tensor], torch.distributions.Distribution]], 
                  differentiable: bool = False, 
                  input_layer: bool = False, 
                  batch_num = [[], []],
                  shape = None,
                  ):
        super().__init__()
        self.dist = dists[0] #function to construct the inference probability distribution (e.g. self.dist([x,y]) makes a normal distribution with mean x and var y)
        self.gen_dist = dists[1] #function to construct a generative probability distribution
        self.input_layer = input_layer #bool to indicate whether or not the layer receives stimulus inputs
        self.differentiable = differentiable #bool to indicate whether rsample or sample should be used (in all networks we use, differentiable=False)
        self.func = None #function to produce the inputs for dist (e.g. self.func(input) = [x,y], then self.dist([x,y]) makes a normal distribution with mean x and var y)
        self.gen_func = None #function to produce the inputs for gen_dist
        self.batch_num = batch_num[0] #number of samples in each inference batch
        self.batch_num_gen = batch_num[1] #number of samples in each generative batch
        self.shape = shape

    def forward(self, x: Tensor, gen = False):
        if not gen:
            self.dist_params = self.calc_dist_params(x)
            self.output = self.sample()
            return self.output
        else:
            self.gen_dist_params = self.calc_dist_params(x, gen = True)
            self.gen_output = self.sample(gen = True)
            return self.gen_output

    def dynamic_mixed_forward(self, x, x_gen, mixing_constant: float, timescale: float, apical_lesion = False, mode = 'interp'):
        """Function for sampling neural activity in a single layer based on a mixture of top-down and bottom-up input.

        Layers are coordinated by the dynamic_mixed_forward method of the InfGenNetwork class

        We use this in the DynamicMixedSampler callback, after a network has been trained, to simulate hallucination.
        NOTE: currently assumes that self.dist and self.gen_dist are both Normal distributions

        Inputs:
        x: bottom-up input
        x_gen: top-down input
        mixing_constant: determines how much bottom-up vs top-down inputs determine network activity
        timescale: determines speed at which network activity evolves through time
        apical_lesion: determines whether top-down inputs are silenced
        mode: options of 'interp' (interpolated sampling), 'noise' (noise-based protocol), 'excitability' (hallucination is modeled as an excitability change)
        """
        self.forward(x)
        self.forward(x_gen, gen= True)
        inf_params = self.dist_params
        gen_params = self.gen_dist_params
        tau = 0.35
        if mode == 'interp':
            if not(apical_lesion):
                mean = tau * torch.log(mixing_constant * torch.exp(inf_params[0]/tau) + (1-mixing_constant) * torch.exp(gen_params[0]/tau))
                var = tau * torch.log(mixing_constant * torch.exp(inf_params[1]/tau) + (1-mixing_constant) * torch.exp(gen_params[1]/tau))
            else:
                if torch.cuda.is_available():
                    mean = tau * torch.log(mixing_constant * torch.exp(inf_params[0]/tau) + (1-mixing_constant) * torch.exp(torch.zeros(gen_params[0].shape, device = torch.cuda.current_device())/tau))
                else:
                    mean = tau * torch.log(mixing_constant * torch.exp(inf_params[0]/tau) + (1-mixing_constant) * torch.exp(torch.zeros(gen_params[0].shape)/tau))
                var = tau * torch.log(mixing_constant * torch.exp(inf_params[1]/tau) + (1-mixing_constant) * torch.exp(gen_params[1]/tau))
        elif mode == 'noise':
            mean = inf_params[0]
            var_slope = 1
            var = inf_params[1] + var_slope * (1-mixing_constant)
        elif mode == 'excitability':
            mean_slope = 1
            mean = inf_params[0] * (1 + mean_slope * (1-mixing_constant))
            var = inf_params[1]

        dist = self.dist([timescale * mean, torch.sqrt(torch.tensor(timescale)) * var])
        output_update = dist.sample()
        self.dynamic_mixed_output_prev = self.dynamic_mixed_output
        self.dynamic_mixed_output = (1-timescale) * self.dynamic_mixed_output_prev + output_update

    def reset_dynamic_mixed_output(self):
        self.dynamic_mixed_output_prev = self.output
        self.dynamic_mixed_output = self.output

    def sample(self, gen: bool = False):
        if not gen:
            dist = self.dist(self.dist_params)
            batch_num = torch.Size(self.batch_num)
        else:
            dist = self.gen_dist(self.gen_dist_params)
            batch_num = torch.Size(self.batch_num_gen)
        if self.differentiable:
            return dist.rsample(sample_shape = batch_num)
        else:
            return dist.sample(sample_shape = batch_num)
        
    def calc_dist_params(self, x: Tensor, gen: bool = False):
        return x

    def log_prob(self, x: Tensor | None = None, output: Tensor | None = None, gen: bool = False):
        """Used to calculate the loss for a single network layer"""
        if ((x is None) or (output is None)):
            if not(gen):
                self.predicted_activity_inf = self.dist_params
                dist = self.dist(self.predicted_activity_inf)
                output = self.output
                
            else:
                self.predicted_activity_gen = self.gen_dist_params
                dist = self.gen_dist(self.predicted_activity_gen)
                output = self.gen_output
        else:
            if not(gen):
                self.predicted_activity_inf = self.calc_dist_params(x)
                dist = self.dist(self.predicted_activity_inf)
            else:
                self.predicted_activity_gen = self.calc_dist_params(x, gen = True)
                dist = self.gen_dist(self.predicted_activity_gen)
        
        return dist.log_prob(output)
    
    def reset(self):
        if self.shape is None:
            self.output = None
            self.gen_output = None
            self.mixed_output = None
        else:
            if torch.cuda.is_available():
                self.output = torch.zeros(self.shape, device = torch.cuda.current_device())
                self.gen_output = torch.zeros(self.shape, device = torch.cuda.current_device())
                self.mixed_output = torch.zeros(self.shape, device = torch.cuda.current_device())
            else:
                self.output = torch.zeros(self.shape)
                self.gen_output = torch.zeros(self.shape)
                self.mixed_output = torch.zeros(self.shape)

    def get_output(self, mixed = False, gen = False):
        if mixed:
            return self.mixed_output
        elif gen:
            return self.gen_output
        else:
            return self.output
    
class IGFuncLayerWrapper(InfGenProbabilityLayer):
    """Constructs an InfGenProbabilityLayer where calc_dist_params is performed using the output of a unique function for each of inference and generation"""
    def __init__(self, 
                 funcs: list[Callable[[Tensor], Tensor]], 
                 dists: list[Callable[[Tensor], torch.distributions.Distribution]], 
                 differentiable: bool = False, 
                 idxs = None, 
                 gen_idxs = None, 
                 batch_num = [[], []],
                 stack = True,
                 gen_stack = True,
                 flatten = False,
                 shape = None):
        #idxs specifies which input indices to process. This is only important if different parts of the input are sent to different areas of the network
        input_layer = not(idxs is None)
        super().__init__(dists, differentiable = differentiable, input_layer = input_layer, batch_num = batch_num, shape = shape)
        self.stack = stack
        self.gen_stack = gen_stack
        self.flatten = flatten
        self.func = funcs["inf"]
        self.gen_func = funcs["gen"]
        self.idxs = idxs #idxs determines which indices of a list of inputs are fed in for inference
        self.gen_idxs = gen_idxs
        

    def calc_dist_params(self, x: Tensor, gen: bool = False):
        if not(gen):
            func = self.func
            if not(self.idxs is None):
                x = [x[idx] for idx in self.idxs]
            if self.flatten:
                x = [element.flatten(start_dim = 1) for element in x]
            if self.stack:
                x = torch.hstack(x)
        else:
            func = self.gen_func
            if not(self.gen_idxs is None):
                x = [x[idx] for idx in self.gen_idxs]
            if self.flatten:
                x = [element.flatten(start_dim = 1) for element in x]
            if self.gen_stack:
                x = torch.hstack(x)
        return func(x)

class View(nn.Module):
    """Basic module for reshaping inputs into a desired shape"""
    def __init__(self, shape: list, no_batch=False):
        super().__init__()
        self.shape = shape #takes a list indicating the non-batch indices to reshape the input into
        self.no_batch = no_batch
    
    def forward(self, x: Tensor):
        x = x
        if not(self.no_batch):
            return x.view([x.shape[0], *self.shape])
        else:
            return x.view([*self.shape])
    
class MeanParams(nn.Module):
    """A nn.Module layer that outputs a stored mean and variance. Used for the top-layer of a generative network"""
    def __init__(self, N: int, batch_size: int, mean_val: float = 0., scale: float = 1.):
        super().__init__()
        self.mean = torch.nn.Parameter(mean_val * torch.ones(batch_size, N), requires_grad = False)
        self.scale = torch.nn.Parameter(scale * torch.ones(N), requires_grad = False)
    def forward(self, x:Tensor) -> Tensor:
        return [self.mean, self.scale]
    
class MeanPlusNoise(nn.Module):
    """A nn.Module layer that is used to parameterize a Normal distribution with an input-dependent mean and a constant variance"""
    def __init__(self, N, mean_func: Callable[[Tensor],Tensor], scale):
        super().__init__()
        self.mean_func = mean_func
        if torch.cuda.is_available():
            self.scale = scale * torch.ones(N, device = torch.cuda.current_device())
        else:
            self.scale = scale * torch.ones(N)
        self.N = N
    
    def forward(self, x:Tensor) -> list[Tensor]:
        return [self.mean_func(x), self.scale]

class MeanExpScale(nn.Module):
    """A nn.Module layer that is used to parameterize a normal distribution with an input-dependent mean and variance, as in the final layer of a VAE"""
    def __init__(self, 
                 mean_func: Callable[[Tensor],Tensor],
                 scale_func: Callable[[Tensor],Tensor], 
                 ):
        super().__init__()
        self.mean_func = mean_func
        self.scale_func = scale_func
    
    def forward(self, x: Tensor) -> list[Tensor]:
        return [self.mean_func(x), torch.exp(self.scale_func(x))]

class BranchedDendrite(nn.Module):
    """The multicompartment dendrite layer used in the main results of the paper"""
    def __init__(self, dim_in, branch_num, dim_out, nl: nn.Module, batch_norm = True):
        super().__init__()
        self.branch_num = branch_num #number of dendritic branches
        self.dim_out = dim_out
        self.lin_1 = nn.Linear(dim_in, branch_num * dim_out) #nn.Linear(dim_in, [branch_num, dim_out])
        self.lin_2 = nn.Linear(branch_num, dim_out)

        #Enforce positivity in lin_2 and lin_3, since these are meant to be conductances, and conductances can't change the sign of the input current
        self.lin_2.weight.data = torch.abs(self.lin_2.weight.data)

        self.batch_norm_bool = batch_norm
        self.batch_norm = nn.BatchNorm1d(branch_num * dim_out, affine = True)
        self.batch_norm_2 = nn.BatchNorm1d(dim_out, affine = False)
        self.nl = nl
        self.nl_2 = nn.Tanh()

    def forward(self, x):
        #Enforce positivity in lin_2, since these are meant to be conductances, and conductances can't change the sign of the input current
        self.lin_2.weight.data = torch.clamp(self.lin_2.weight.data, min = 0)

        #First set of dendritic branches
        x = self.lin_1(x)

        x = self.batch_norm(self.nl(x).flatten(start_dim = 1)).view([x.shape[0], self.dim_out, self.branch_num])

        #second set of dendritic branches
        x = torch.sum(x * self.lin_2.weight, dim = -1) + self.lin_2.bias

        #final nonlinearity
        x = self.nl_2(x).flatten(start_dim = 1)
        if self.batch_norm_bool:
            x = self.batch_norm_2(x)
        return x

class DiffusionInf(nn.Module):
    """Simple layer in our recurrent network model that replaces part of the layer state x with noise"""
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return np.sqrt(1 - self.beta) * x
    
class DiffusionGenNL(nn.Module):
    """Nonlinearity for implementing a T timesteps of recurrent processing in a given layer. We take T=1 for all results"""
    def __init__(self, N, T):
        super().__init__()
        self.N = N
        self.T = T
        self.T_ctr = 0
        self.lin = nn.Linear(N,N)
        self.lin2 = nn.Linear(N,N)

        self.nl = nn.Tanh()
        self.nl2 = nn.Sigmoid()

    def forward(self, x):
        if self.T_ctr == 0:
            self.state_prev = x
        else:
            self.state_prev = self.state
        self.gate = self.nl2(self.lin2(x))

        self.state = self.state_prev + self.gate*self.nl(self.lin(x))
        if self.T_ctr == self.T - 1:
            self.T_ctr = 0
        else:
            self.T_ctr += 1
        return self.state

class DiffusionGen(nn.Module):
    """Essentially just functions as a nonlinearity layer"""
    def __init__(self, nl, beta):
        super().__init__()
        self.nl = nl
        self.beta = beta

    def forward(self, x):
        return self.nl(x)