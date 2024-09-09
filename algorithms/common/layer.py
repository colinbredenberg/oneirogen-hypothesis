"""
Classes for defining individual layers
"""
from __future__ import annotations
from typing import Callable
import torch
from torch import autograd, nn, Tensor
import torch.nn.functional as F
import math
from diffusers import AutoencoderTiny
from collections import OrderedDict
from torch.distributions.log_normal import LogNormal
import pyrtools as pt
import plenoptic as po
from plenoptic.simulate import SteerablePyramidFreq
from plenoptic.synthesize import Eigendistortion
from plenoptic.tools.data import to_numpy
import numpy as np

# def diagonal_KL(mean_0, std_0, mean_1, std_1):
#     """Calculates the KL divergence between two diagonal normal distributions (D_KL(N_0||N_1))"""
#     if std_0.dim() > 1:
#         k = torch.numel(std_0[0,...])
#     else:
#         k = torch.numel(std_0)
#     D_KL = 1/2 * (torch.sum(std_0**2/std_1**2 + (mean_1 - mean_0)**2/std_1**2 + torch.log(std_1) - torch.log(std_0), axis = -1) - k)
#     return D_KL

class Layer(nn.Module):
    """ These layers are designed to be combined into Networks. These layers extend nn.Module to calculate local gradients for bio-plausible learning

    Attributes
    ----------
    x: the layer's inputs

    Methods
    ----------
    forward:
        the layer processes its inputs
    local_grad:
        Combines a vector grad_vec which contains the derivative of the loss with respect to the output
        with the derivative of the output with respect to the input variable.
    """
    def __init__(self):
        """Basic initialization"""
        super().__init__()
        self.output: Tensor | None = None
        self.x:Tensor

    def forward(self,x: Tensor):
        """the layer processes its inputs"""
        self.x = x
        pass

    def local_grad(self, grad_vec: Tensor, input: Tensor) -> tuple[Tensor, ...]:
        """
        Combines a vector grad_vec which contains the derivative of the loss with respect to the output
        with the derivative of the output with respect to the input variable.
        Output: 1xK gradient tensor, where K is the input's dimension
        """
        return autograd.grad(torch.dot(grad_vec.flatten(), self.output.flatten()), input, retain_graph = True)

class ClassificationRLLoss(Layer):
    """Extends Layer to compute a reward based on the equality of the network output and its target"""
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        self.output = (x == y).float()
        return self.output
        
class LayerWrapper(Layer):
    """
    Takes a torch.nn.Module and adapts it to comply with the Layer class
    """
    def __init__(self, module: nn.Module):
        super().__init__()
        assert isinstance(module,nn.Module)
        self.module = module
    
    def forward(self,x: tuple[Tensor,...]):
        self.output = self.module(*x)
        return self.output

class FuncLayerWrapper(Layer):
    """
    Takes a torch.nn.Module and a function (or composition of functions) and adapts to comply with the Layer class
    """
    def __init__(self, module: nn.Module, function: Callable[[Tensor], Tensor]):
        super().__init__()
        assert isinstance(module, nn.Module)
        self.module = module
        self.function = function
    
    def forward(self, x: Tensor) -> Tensor:
        self.output = self.function(self.module(*x))
        return self.output

class ProbabilityLayer(nn.Module):
    """
    Takes a torch.distributions distribution and adapts it to function somewhat like a nn.Module
    example: self.categorical = ProbabilityWrapper(lambda x: torch.distributions.categorical.Categorical(x))

    Attributes
    ----------
    x: the layer's inputs
    dist: the network's distribution
    differentiable: Boolean indicating whether or not the network should register its sampling with autograd
    dist_params: inputs necessary for sampling from self.dist
    output: the sampled output of the probability layer

    Methods
    ----------
    forward:
        the layer processes its inputs, sets dist_params, and sets output
    sample:
        the layer produces a data sample depending on its distribution parameters
    log_prob:
        calculates the log probability of the layer's output sample wrt its own probability distribution
    """
    def __init__(self, dist: Callable[[Tensor], torch.distributions.Distribution], differentiable: bool = False, shape = None):
        super().__init__()
        self.dist = dist
        self.differentiable = differentiable
        self.shape = shape

    def forward(self, x: Tensor) -> Tensor:
        self.dist_params = x
        self.output = self.sample()
        return self.output

    def sample(self) -> Tensor:
        if self.differentiable:
            return self.dist(self.dist_params).rsample()
        else:
            return self.dist(self.dist_params).sample()
    
    def log_prob(self) -> Tensor:
        return self.dist(self.dist_params).log_prob(self.output)
    
    def reset(self):
        if self.shape is None:
            self.output = None
            self.gen_output = None
            self.mixed_output = None
            self.dynamic_mixed_output = None
        else:
            self.output = torch.zeros(self.shape)
            self.gen_output = torch.zeros(self.shape)
            self.mixed_output = torch.zeros(self.shape)
            self.dynamic_mixed_output = torch.zeros(self.shape)
            
    def get_output(self, mixed = False, gen = False):
        if mixed:
                return self.mixed_output
        elif gen:
            return self.gen_output
        else:
            return self.output

    

class ProbModuleLayer(ProbabilityLayer):
    """Extends the ProbabilityLayer class so that a layer's distribution parameters are calculated by a nn.Module
    """
    def __init__(self, dist: Callable[[Tensor], torch.distributions.Distribution], module: nn.Module, differentiable = False):
        super().__init__(dist, differentiable = differentiable)
        self.module = module
    
    def forward(self, x: Tensor) -> Tensor:
        self.dist_params = self.calculate_dist_params(x)
        self.output = self.sample()
        return self.output
    
    def calculate_dist_params(self, x:Tensor) -> Tensor:
            x = torch.hstack(x)
            return self.module.forward(x)
    
class InfGenProbabilityLayer(nn.Module):
    """
    Similar to ProbabilityLayer, except that it defines an inference distribution and a generative distribution,
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
                  is_t0_layer: bool = True,
                  is_transition_layer: bool = True
                  ):
        super().__init__()
        self.dist = dists[0]
        self.gen_dist = dists[1]
        self.input_layer = input_layer
        self.differentiable = differentiable
        self.func = None
        self.gen_func = None
        self.batch_num = batch_num[0]
        self.batch_num_gen = batch_num[1]
        self.shape = shape
        self.is_t0_layer = is_t0_layer
        self.is_transition_layer = is_transition_layer

    def forward(self, x: Tensor, gen = False):
        if not gen:
            self.dist_params = self.calc_dist_params(x)
            self.output = self.sample()
            return self.output
        else:
            self.gen_dist_params = self.calc_dist_params(x, gen = True)
            self.gen_output = self.sample(gen = True)
            return self.gen_output
        
    def mixed_forward(self, x: Tensor, mixing_constant: float):
        self.forward(x, gen = True)
        if self.dist is LogNormal:
            self.mixed_output = self.output**(mixing_constant) * self.gen_output**(1-mixing_constant)
        else:
            # self.mixed_output = mixing_constant * self.output + (1-mixing_constant) * self.gen_output
            # mixed_sign = mixing_constant * torch.sign(self.output) + (1-mixing_constant) * torch.sign(self.gen_output)
            # mixed_val = torch.abs(self.output)**(mixing_constant) * torch.abs(self.gen_output)**(1-mixing_constant)
            # self.mixed_output = mixed_sign * mixed_val
            tau = 0.2
            self.mixed_output = tau * torch.log(mixing_constant * torch.exp(self.output/tau) + (1-mixing_constant) * torch.exp(self.gen_output/tau))

    def dynamic_mixed_forward(self, x, x_gen, mixing_constant: float, timescale: float, geometric_mean = False, apical_lesion = False, mode = 'interp'):
        self.forward(x)
        self.forward(x_gen, gen= True)
        inf_params = self.dist_params
        gen_params = self.gen_dist_params
        if geometric_mean:
            mean = torch.relu(inf_params[0])**(mixing_constant) * torch.relu(gen_params[0])**(1-mixing_constant)
            var = torch.relu(inf_params[1])**(mixing_constant) * torch.relu(gen_params[1])**(1-mixing_constant)
        else:
            # mixed_sign = mixing_constant * torch.sign(inf_params[0]) + (1-mixing_constant) * torch.sign(gen_params[0])
            # mixed_val = torch.abs(inf_params[0])**(mixing_constant) * torch.abs(gen_params[0])**(1-mixing_constant)
            # mean = mixed_sign * mixed_val
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
                 shape = None,
                 is_t0_layer: bool = True,
                 is_transition_layer: bool = True):
        #idxs specifies which input indices to process. This is only important if different parts of the input are sent to different areas of the network
        input_layer = not(idxs is None)
        super().__init__(dists, differentiable = differentiable, input_layer = input_layer, batch_num = batch_num, shape = shape, is_t0_layer = is_t0_layer, is_transition_layer = is_transition_layer)
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

class StandardNormalIGFuncLayerWrapper(IGFuncLayerWrapper):
    def standard_normal_KL_divergence(self):
        mean = self.dist_params[0]
        var = self.dist_params[1]
        KL = 1/2 * (torch.sum(var, axis = -1) - torch.sum(torch.log(var), axis = -1) - mean.shape[-1] + torch.sum(mean**2, axis = 1)) #TODO: fix the dims on this
        return KL

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
    
    
class NormalParams(nn.Module):
    """Module for outputting a stored mean and covariance parameter set
    Parameters:
        mean: initialized as a 1xN zero vector
        prec: Initialized as an NxN identity matrix.
    """
    def __init__(self, N: int, mod_mean: Tensor | None = None, mod_prec: Tensor | None = None, eps: float = 1e-6):
        super().__init__()
        self.mod = mod_mean
        self.mod_prec = mod_prec
        self.mean = torch.nn.Parameter(torch.zeros(N), requires_grad = False)
        self.prec = torch.nn.Parameter(torch.normal(torch.zeros(N), 1), requires_grad = True) #torch.nn.Parameter(torch.eye(N))
        self.eps: Tensor  # for type-checking
        self.register_buffer("eps", eps * torch.eye(N), persistent=True)
        self.eye: Tensor
        self.register_buffer("eye", torch.eye(N), persistent=True)
        #self.dynamic_range = torch.nn.Parameter(torch.normal(torch.zeros(N),1))
        self.alpha = 0.1
        # self.mean = torch.nn.Parameter(torch.zeros(N), requires_grad = False)
        # self.prec = torch.nn.Parameter(torch.normal(torch.zeros(N), 1), requires_grad = True) #torch.nn.Parameter(torch.eye(N))
        # self.eps = eps * torch.eye(N)
        # self.eye = torch.eye(N)
        # #self.dynamic_range = torch.nn.Parameter(torch.normal(torch.zeros(N),1))
        # self.alpha = 0.1

    def forward(self, x: Tensor) -> list[Tensor]:
        if self.mod is None:
            mean = self.mean
        else:
            mean = self.mod(x[:,0:-1])
        
        if self.mod_prec is None:
            prec = self.prec
            prec_weighting = self.eye
        else:
            prec = 100/(F.softplus(-torch.log(x[:,[-1]])))
            prec = prec.view([prec.shape[0],1,1])
            prec_weighting = torch.diag_embed(self.mod_prec(-torch.log(x[:,[-1]])))

        return [mean, ((self.alpha * self.eye + (1-self.alpha) * prec_weighting) * prec)]
    
class MeanParams(nn.Module):
    def __init__(self, N: int, batch_size: int, mean_val: float = 0., scale: float = 1.):
        super().__init__()
        self.mean = torch.nn.Parameter(mean_val * torch.ones(batch_size, N), requires_grad = False)
        self.scale = torch.nn.Parameter(scale * torch.ones(N), requires_grad = False)
    def forward(self, x:Tensor) -> Tensor:
        return [self.mean, self.scale]
    
class ConvMeanParams(nn.Module):
    def __init__(self, shape: list, batch_size: int, mean_val: float = 0., scale: float = 1.):
        super().__init__()
        self.mean = torch.tensor(mean_val, device = torch.cuda.current_device())
        self.scale = torch.tensor(scale, device = torch.cuda.current_device())
        self.batch_size = torch.tensor(batch_size, device = torch.cuda.current_device())
        self.shape = torch.tensor(shape, device = torch.cuda.current_device())
        # self.mean = torch.nn.Parameter(mean_val * torch.ones(batch_size, *shape), requires_grad = False)
        # self.scale = torch.nn.Parameter(scale * torch.ones(*shape), requires_grad = False)

    def forward(self, x:Tensor) -> Tensor:
        base_tensor = torch.ones(self.batch_size, *self.shape, device = torch.cuda.current_device())
        return [self.mean * base_tensor, self.scale * base_tensor]
    
class ConvGammaParams(nn.Module):
    def __init__(self, shape: list[int], batch_size: int, concentration_val: float = 1., rate_val: float = 1.):
        super().__init__()
        self.concentration = torch.nn.Parameter(concentration_val * torch.ones(batch_size, *shape), requires_grad = True)
        self.rate_val = torch.nn.Parameter(rate_val * torch.ones(batch_size, *shape), requires_grad = True)

    def forward(self, x:Tensor) -> Tensor:
        return [self.concentration, self.rate_val]
    
class MeanPlusNoise(nn.Module):
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

class ExpScaling(nn.Module):
    def __init__(self, N_in, N):
        super().__init__()
        self.lin_1 = nn.Linear(N_in, N)
        self.lin_2 = nn.Linear(N_in, N)

    def forward(self, x:Tensor) -> Tensor:
        return torch.exp(torch.tanh(self.lin_1(x))) * self.lin_2(x)
    
class Exp(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:Tensor) -> Tensor:
        return torch.sigmoid(torch.exp(x))
    
class ConvMeanPlusNoise(nn.Module):
    def __init__(self, size, mean_func: Callable[[Tensor], Tensor], scale, idx = None):
        super().__init__()
        self.mean_func = mean_func
        if torch.cuda.is_available():
            self.scale = scale * torch.ones(*size, device = torch.cuda.current_device())
        else:
            self.scale = scale * torch.ones(*size)
        self.size = size
        self.idx = idx
    
    def forward(self, x:Tensor) -> list[Tensor]:
        if not (self.idx is None):
            return [self.mean_func(x)[self.idx], self.scale]
        else:
            return [self.mean_func(x), self.scale]

class LinearVariance(nn.Module):
    def __init__(self, mean_func: Callable[[Tensor], Tensor], scale, epsilon = 1e-2, idx = None):
        super().__init__()
        self.mean_func = mean_func
        self.scale = scale
        self.idx = idx
        self.epsilon = epsilon

    def forward(self, x:Tensor) -> list[Tensor]:
        if not (self.idx is None):
            mean_val = self.mean_func(x[self.idx])
        else:
            mean_val = self.mean_func(x)

        return [mean_val, self.scale * torch.sqrt(torch.abs(mean_val)) + self.epsilon]
        

class MeanScale(nn.Module):
    def __init__(self, 
                 mean_func: Callable[[Tensor],Tensor],
                 scale_func: Callable[[Tensor],Tensor], 
                 epsilon: float = 0.001,
                 ):
        super().__init__()
        self.mean_func = mean_func
        self.scale_func = scale_func
        self.epsilon = epsilon
    
    def forward(self, x: Tensor) -> list[Tensor]:
        return [self.mean_func(x), self.scale_func(x) + self.epsilon]

class MeanExpScale(nn.Module):
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
    def __init__(self, dim_in, branch_num, dim_out, nl: nn.Module, batch_norm = True):
        super().__init__()
        self.branch_num = branch_num
        self.dim_out = dim_out
        self.lin_1 = nn.Linear(dim_in, branch_num * dim_out) #nn.Linear(dim_in, [branch_num, dim_out])
        self.lin_2 = nn.Linear(branch_num, dim_out)

        #Enforce positivity in lin_2 and lin_3, since these are meant to be resistances, and resistances can't change the sign of the input current
        self.lin_2.weight.data = torch.abs(self.lin_2.weight.data)
        # self.lin_3.weight.data = torch.abs(self.lin_3.weight.data)
        self.batch_norm_bool = batch_norm
        self.batch_norm = nn.BatchNorm1d(branch_num * dim_out, affine = True)
        self.batch_norm_2 = nn.BatchNorm1d(dim_out, affine = False)
        self.nl = nl
        self.nl_2 = nn.Tanh()

    def forward(self, x):
        #Enforce positivity in lin_2 and lin_3, since these are meant to be resistances, and resistances can't change the sign of the input current
        self.lin_2.weight.data = torch.clamp(self.lin_2.weight.data, min = 0)
        # self.lin_3.weight.data = torch.clamp(self.lin_3.weight.data, min = 0)

        #First set of dendritic branches
        x = self.lin_1(x)
        # x = x.view([x.shape[0], self.dim_out, self.branch_num])
        # if self.batch_norm_bool:
        x = self.batch_norm(self.nl(x).flatten(start_dim = 1)).view([x.shape[0], self.dim_out, self.branch_num])
        # x = self.nl(x).flatten(start_dim = 1).view([x.shape[0], self.dim_out, self.branch_num])

        #second set of dendritic branches
        x = torch.sum(x * self.lin_2.weight, dim = -1) + self.lin_2.bias#self.lin_2(x)
        #final nonlinearity
        x = self.nl_2(x).flatten(start_dim = 1)
        if self.batch_norm_bool:
            x = self.batch_norm_2(x)
        return x

class SteerablePyramid(nn.Module):
    def __init__(self,
                 imshape,
                 order,
                 scales,
                 is_complex=True,
                 is_polar=False,
                 stack=True):
        super().__init__()
        self.imshape = imshape
        self.order = order
        self.scales = scales
        self.is_complex = is_complex
        self.is_polar = is_polar
        self.stack = stack
        self.pyr = SteerablePyramidFreq(height=self.scales,image_shape=self.imshape[1::],
                                          order=self.order,is_complex = self.is_complex,twidth=1, downsample=False).to(torch.cuda.current_device())

    def forward(self, x: Tensor) -> list[Tensor]:
        out = self.pyr(x)
        if self.stack:
            out, _ = self.pyr.convert_pyr_to_tensor(out)
            if self.is_complex and not(self.is_polar):
                out_re = out.real
                out_im = out.imag
                out = torch.cat([out_re, out_im], dim = 1)
            elif(self.is_complex and self.is_polar):
                out_abs = torch.abs(out)
                out_angle = torch.angle(out)
                out = torch.cat([out_abs, torch.sin(out_angle), torch.cos(out_angle)], dim = 1)
            else:
                out = out
        else:
            if self.is_complex:
                for key, value in out.items():
                    value_re = value.real
                    value_im = value.imag
                    out[key] = torch.cat([value_re, value_im])
        return out

class InvSteerablePyramid(nn.Module):
    def __init__(self,
                 imshape,
                 order,
                 scales,
                 is_complex=True,
                 is_polar=False,
                 stack = True):
        super().__init__()
        self.imshape = imshape
        self.order = order
        self.scales = scales
        self.is_complex = is_complex
        self.is_polar = is_polar
        self.stack = stack
        self.pyr = SteerablePyramidFreq(height=self.scales,image_shape=self.imshape[1::],
                                          order=self.order,is_complex = self.is_complex,twidth=1, downsample=False).to(torch.cuda.current_device())

        test_image = torch.zeros(12, *self.imshape).to(torch.cuda.current_device())
        out = self.pyr(test_image)
        out, self.pyr_info = self.pyr.convert_pyr_to_tensor(out)

    def forward(self, x: Tensor) -> list[Tensor]:
        if self.stack:
            if self.is_complex and not(self.is_polar):
                half_idx = int(x.shape[1]/2)
                out = torch.complex(x[:,0:half_idx,...], x[:,half_idx::,...])
            elif self.is_complex and self.is_polar:
                third_idx = int(x.shape[1]/3)
                angle = torch.arctan(x[:,third_idx:(2*third_idx)]/x[:,(2*third_idx):(3*third_idx)])
                out = torch.polar(x[:,0:third_idx,:], angle)
            else:
                out = x
            out = self.pyr.convert_tensor_to_pyr(out, *self.pyr_info)
        else:
            if self.is_complex:
                for key, value in out.items():
                    half_idx = int(value.shape[1]/2)
                    out[key] = torch.complex(value[:,0:half_idx,...], x[:,half_idx::,...])
        out = self.pyr.recon_pyr(out)
        return out

class TinyAutoencoder(nn.Module):
    def __init__(self,
                 requires_grad=False):
        super().__init__()
        self.encoder = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype = torch.float32).encoder.to(torch.cuda.current_device())
        self.encoder.requires_grad_(requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)
    
class TinyAutodecoder(nn.Module):
    def __init__(self, requires_grad = False):
        super().__init__()
        self.decoder = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype = torch.float32).decoder.to(torch.cuda.current_device())
        self.decoder.requires_grad_(requires_grad)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(x)

class DictConv2d(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride=1,
                 padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: list[dict]) -> Tensor:
        x = x[0]
        for key, value in x.items():
            x[key] = self.conv(value)
        return x

class DictConvTranspose2d(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size,
                 stride=1,
                 padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x: list[dict]) -> Tensor:
        x = x[0]
        for key, value in x.items():
            x[key] = self.conv_transpose(value)
        return x
    
class DictFlatten(nn.Module):
    def __init__(self, start_dim):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: list[dict]) -> Tensor:
        x = x[0]
        x_list = []
        for value in x.values():
            x_list.append(torch.flatten(value, start_dim = self.start_dim))
        return torch.hstack(x_list)
    
class DictUnflatten(nn.Module):
    def __init__(self, size_list, keys):
        super().__init__()
        self.size_list = size_list
        self.keys = keys
    
    def forward(self, x: Tensor) -> dict:
        ret_dict = OrderedDict()
        start_idx = 0
        for key, size in zip(self.keys, self.size_list):
            length = torch.prod(size)
            ret_dict[key] = torch.unflatten(x[:,start_idx:(start_idx + length)], size)
        return ret_dict
    
class PolarNL(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x:Tensor) -> Tensor:
        third_idx = int(x.shape[1]/3)
        out_0 = np.pi * F.sigmoid(x[:,0:third_idx,...])
        out_1 = F.tanh(x[:,third_idx::,...])#torch.remainder(x[:,0:half_idx,...], np.pi)
        return torch.cat([out_0, out_1], dim = 1)

class Transform(nn.Module):
    def __init__(self,
                 transform: Callable[[Tensor], Tensor]):
        super().__init__()
        self.transform = transform

    def forward(self, x: Tensor) -> list[Tensor]:
        return self.transform(x)

class DiffusionInf(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return np.sqrt(1 - self.beta) * x
    
class RecurrentLayer(nn.Module):
    def __init__(self, N, T):
        super().__init__()
        self.N = N
        self.T = T
        self.T_ctr = 0
        self.lin = nn.Linear(N,N)
        self.nl = nn.Tanh()

    def forward(self, x):
        if self.T_ctr == 0:
            self.state_prev = x
        else:
            self.state_prev = self.state
        self.state = self.state_prev + self.nl(self.lin(x))
        if self.T_ctr == self.T - 1:
            self.T_ctr = 0
        else:
            self.T_ctr += 1
        return self.state
    

class DiffusionGenNL(nn.Module):
    def __init__(self, N, T):
        super().__init__()
        self.N = N
        self.T = T
        self.T_ctr = 0
        self.lin = nn.Linear(N,N)
        self.lin2 = nn.Linear(N,N)
        # self.lin3 = nn.Linear(N,N)
        self.nl = nn.Tanh()
        self.nl2 = nn.Sigmoid()

    def forward(self, x):
        if self.T_ctr == 0:
            self.state_prev = x
        else:
            self.state_prev = self.state
        self.gate = self.nl2(self.lin2(x))
        # self.attention = self.nl2(self.lin3(x))
        self.state = self.state_prev + self.gate*self.nl(self.lin(x))
        if self.T_ctr == self.T - 1:
            self.T_ctr = 0
        else:
            self.T_ctr += 1
        return self.state #self.lin_2(self.nl(self.lin(x)))

class DiffusionGen(nn.Module):
    def __init__(self, nl, beta):
        super().__init__()
        self.nl = nl
        self.beta = beta

    def forward(self, x):
        return self.nl(x)

class ConvDiffusionGen(nn.Module):
    def __init__(self, n_channels, kernel_size, padding, T, beta):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        # self.lin_2 = nn.Linear(N_int, N)
        self.nl = nn.Tanh()
        self.beta = beta
        self.T = T
        self.T_ctr = 0

    def forward(self, x):
        if self.T_ctr == 0:
            self.state_prev = x
        else:
            self.state_prev = self.state
        self.state = self.state_prev + self.nl(self.conv(x))
        if self.T_ctr == self.T - 1:
            self.T_ctr = 0
        else:
            self.T_ctr += 1
        return self.state

class ConvNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.inf_network = nn.Sequential(ConvMLPBlock(in_channel, 64, 64, 9, stride = 1, padding = 0),
                                         ConvMLPBlock(64, 64, 64, 9, stride = 1, padding = 0),
                                         ConvMLPBlock(64, 64, 64, 9, stride = 1, padding = 0),
                                         ConvMLPBlock(64, 64, 64, 5, stride = 1, padding = 0),
                                         ConvMLPBlock(64, 64, 64, 4, stride = 1, padding = 0)
                                    )
    def forward(self,x):
        return self.inf_network(x)
    
class InvConvNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.gen_network = nn.Sequential(
                                         InvConvMLPBlock(64, 64, 64, 9, stride = 1, padding = 0),
                                         InvConvMLPBlock(64, 64, 64, 9, stride = 1, padding = 0),
                                         InvConvMLPBlock(64, 64, 64, 9, stride = 1, padding = 0),
                                         InvConvMLPBlock(64, 64, 64, 5, stride = 1, padding = 0),
                                         InvConvMLPBlock(in_channel, 64, 64, 4, stride = 1, padding = 0),
                                         )
    def forward(self, x):
        return self.gen_network(x)

class ConvNormBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_channel, kernel_size, norm_kernel_size, stride = 1, norm_stride = 1, padding = 0):
        super().__init__()
        self.f1 =nn.Conv2d(in_channel, out_channel, kernel_size, stride = stride, padding = padding)
        self.f2 =nn.Conv2d(in_channel, out_channel, kernel_size, stride = stride, padding = padding)
        self.norm_network = nn.Conv2d(in_channel, norm_channel, norm_kernel_size, stride = norm_stride)
        self.remap_network = nn.ConvTranspose2d(norm_channel, out_channel, 1, stride = norm_stride)
        self.epsilon = 0.001
    def forward(self, x):
        f1 = self.f1(x)
        f2 = self.f2(x)
        amplitudes = torch.sqrt(f1**2 + f2**2)
        phases = torch.atan(f1/f2)
        return phases, amplitudes
    
class InvConvNormBlock(nn.Module):
    def __init__(self, in_channel, out_channel, norm_channel, kernel_size, norm_kernel_size, stride = 1, norm_stride = 1, padding = 0):
        super().__init__()
        self.inv_f1 = nn.ConvTranspose2d(out_channel, in_channel, kernel_size, stride = stride, padding = padding)
        self.inv_f2 = nn.ConvTranspose2d(out_channel, in_channel, kernel_size, stride = stride, padding = padding)
        
    def forward(self, x):
        amplitudes = x[0]
        phases = x[1]
        f1 = amplitudes * torch.sin(phases)
        f2 = amplitudes * torch.cos(phases)
        return self.inv_f1(f1) + self.inv_f2(f2)

class MultiScaleConvBlock(nn.Module):
    def __init__(self, kernel_sizes, in_channels, out_channels):
        super().__init__()
        assert all([kernel_size % 2 == 1 for kernel_size in kernel_sizes])
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        module_list = [nn.Conv2d(in_channels, out_channels, kernel_size, stride = 1, padding = int((kernel_size -1)/2)) for kernel_size in kernel_sizes]
        self.module_list = nn.ModuleList(modules = module_list)

    def forward(self, x):
        outputs = [module(x) for module in self.module_list]
        return torch.cat(outputs, dim = 1)
    
class InvMultiScaleConvBlock(nn.Module):
    def __init__(self, kernel_sizes, in_channels, out_channels):
        super().__init__()
        assert([kernel_size % 2 ==1 for kernel_size in kernel_sizes])
        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        module_list = [nn.ConvTranspose2d(out_channels, in_channels, kernel_size, stride = 1, padding = int((kernel_size - 1)/2)) for kernel_size in kernel_sizes]
        self.module_list = nn.ModuleList(modules = module_list)
    
    def forward(self,x):
        split_x = torch.chunk(x, len(self.kernel_sizes), dim = 1)
        outputs = []
        for idx, input in enumerate(split_x):
            outputs.append(self.module_list[idx](input))
        return torch.sum(torch.stack(outputs), dim = 0)
    
class ConvMLPBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.inf_network = nn.Sequential(nn.Conv2d(in_channel, hidden_channel, kernel_size, stride = stride, padding = padding),
                                         nn.Tanh(),
                                         nn.Conv2d(hidden_channel, out_channel, 1, stride = 1),
                                         nn.Tanh()
        )
    def forward(self, x):
        return self.inf_network(x)
    
class InvConvMLPBlock(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, kernel_size, stride = 1, padding = 0):
        super().__init__()
        self.gen_network = nn.Sequential(nn.ConvTranspose2d(out_channel, hidden_channel, 1, stride = 1),
                                         nn.Tanh(),
                                         nn.ConvTranspose2d(hidden_channel, in_channel, kernel_size, stride = stride, padding = padding),
                                         nn.Tanh(),
        )
    def forward(self,x):
        return self.gen_network(x)

# class MultiScaleConv(nn.Module):
#     def __init__(self, in_channel, out_channel):
#         super().__init__()
#         self.conv1 = nn.Conv2d(out_channel, in_channel, 5, stride = 1)
#         self.conv12 = nn.Conv2d(in_channel, 32, )
#         self.conv2 = nn.Conv2d(out_channel, in_channel, 13, stride = 1)
#         self.nl = nn.Tanh()
    
#     def forward(self, x):
#         x_1 = self.conv1(x)
#         x_2 = self.conv2(x)
        
class LinearGrad(autograd.Function):
    """
    Autograd Function that does a backward pass using the weight_backward matrix of the layer
    Imported from Biotorch
    """
    def forward(context, input, weight, weight_backward, bias=None, bias_backward=None):
        context.save_for_backward(input, weight, weight_backward, bias, bias_backward)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_backward, bias, bias_backward = context.saved_tensors
        grad_input = grad_weight = grad_weight_backward = grad_bias = grad_bias_backward = None
        # Gradient input
        if context.needs_input_grad[0]:
            # Use the weight_backward matrix to compute the gradient
            grad_input = grad_output.mm(weight_backward)
        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        # Gradient bias
        if bias is not None and context.needs_input_grad[3]:
                grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_backward, grad_bias, grad_bias_backward

class FALinear(nn.Linear):
    """Imported from Biotorch"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.weight_backward = nn.Parameter(torch.Tensor(self.weight.size()))
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            self.register_parameter("bias", None)
            self.bias_backward = None

        self.init_parameters()

    def init_parameters(self):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_backward, a=math.sqrt(5))
        # Scaling factor is the standard deviation of Kaiming init.
        self.scaling_factor = 1 / math.sqrt(3 * fan_in)
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            nn.init.uniform_(self.bias_backward, -bound, bound)

    def forward(self,x):
        return LinearGrad.apply(x, self.weight, self.weight_backward, self.bias, self.bias_backward)


class Conv2dGrad(autograd.Function):
    """
    Autograd Function that Does a backward pass using the weight_backward matrix of the layer
    Imported from Biotorch
    """
    @staticmethod
    def forward(context, input, weight, weight_backward, bias, bias_backward):
        context.save_for_backward(input, weight, weight_backward, bias, bias_backward)
        output = torch.nn.functional.conv2d(input,weight)
        return output

    def backward(context, grad_output):
        input, weight, weight_backward, bias, bias_backward = context.saved_tensors
        grad_input = grad_weight = grad_weight_backward = grad_bias = grad_bias_backward = None

        # Gradient input
        if context.needs_input_grad[0]:
            # Use the FA constant weight matrix to compute the gradient
            grad_input = torch.nn.grad.conv2d_input(input_size=input.shape,
                                                    weight=weight_backward,
                                                    grad_output=grad_output)

        # Gradient weights
        if context.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input=input,
                                                      weight_size=weight_backward.shape,
                                                      grad_output=grad_output)

        # Gradient bias
        if bias is not None and context.needs_input_grad[3]:
            grad_bias = grad_output.sum(0).sum(2).sum(1)

        # Return the same number of parameters
        return grad_input, grad_weight, grad_weight_backward, grad_bias, grad_bias_backward, None, None, None, None

class FAConv2d(nn.Conv2d):
    """Imported from Biotorch"""
    def __init__(self,in_channels: int,out_channels: int,kernel_size:int):
        super().__init__(in_channels,out_channels,kernel_size)

        self.weight_backward = nn.Parameter(torch.Tensor(self.weight.size()))
        if self.bias is not None:
            self.bias_backward = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            self.register_parameter("bias", None)
            self.bias_backward = None

        self.init_parameters()

    def init_parameters(self):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_backward, a=math.sqrt(5))
        # Scaling factor is the standard deviation of Kaiming init.
        self.scaling_factor = 1 / math.sqrt(3 * fan_in)
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            nn.init.uniform_(self.bias_backward, -bound, bound)

    def forward(self, x):
        return Conv2dGrad.apply(x,self.weight,self.weight_backward,self.bias,self.bias_backward)