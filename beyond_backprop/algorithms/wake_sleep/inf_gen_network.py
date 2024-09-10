from __future__ import annotations

import torch
from torch import nn, Tensor

import torch.nn.functional as F

import beyond_backprop.algorithms.common.layer as layer
import networkx as nx
from networkx import DiGraph

class InfGenNetwork(nn.Module):
    """
    Define a network that is composed of two directed acyclic graphs of layers: the first defining the forward operation for inference
    and the second defining the forward operation for generation. Requires that each layer be an InfGenProbabilityLayer and that the graphs have identical nodes

    Attributes
    ----------
    graph: nx.Digraph
        DAG that specifies the order of computation for inference
    gen_graph: nx.Digraph
        DAG that specifies the order of computation for generation
    gen_subgraph: nx.Digraph
        DAG that specifies the order of computation for generating activations for a subset of the nodes in the gen_graph
    ts: list
        List specifying an iterable ordering of nodes such that the required inputs from parents are always available for a child node for inference
    gen_ts: list
        List specifying an iterable ordering of nodes such that the required inputs from parents are always available for a child node for generation
    gen_ts_subgraph: list
        List specifying an iterable ordering of nodes such that the required inputs from parents are always available for a child node for generating a subset of the nodes
    inf_group: nn.ModuleList
        Module that can be used to process all of the inference parameters separately from the generative parameters
        e.g. w/ inf_group.parameters()
    gen_group: nn.ModuleList
        Module that can be used to process all of the generative parameters separately from the inference parameters
        e.g. w/ gen_group.parameters()
        
    Methods
    -------
    forward:
        Iterate through self.ts to calculate the activations at each layer in response to inputs
    gen_forward:
        Iterate through self.gen_ts to calculate the activations at each layer to generate sample outputs
    log_prob:
        Iterate through self.ts to calculate the inference log probabilities at each layer wrt the generative samples
    gen_log_prob:
        Iterate through self.gen_ts to calculate the generative log probabilities at each layer wrt the inference samples
    """
    def __init__(self, **kwargs):
        super().__init__()
        if "graphs" in kwargs:
            assert all(isinstance(graph, nx.DiGraph) for graph in kwargs["graphs"])
            assert all(nx.is_directed_acyclic_graph(graph) for graph in kwargs["graphs"])
            self.graph = kwargs["graphs"][0]
            self.gen_graph = kwargs["graphs"][1]

            assert set(self.graph.nodes) == set(self.gen_graph.nodes)
            assert all(isinstance(node, layer.InfGenProbabilityLayer) for node in self.graph.nodes)
        else:
            self.graph = nx.DiGraph()
            self.gen_graph = nx.DiGraph()
        
        self.ts = list(nx.topological_sort(self.graph))
        self.gen_ts = list(nx.topological_sort(self.gen_graph))

        if "t0_graphs" in kwargs:
            assert all(isinstance(graph, nx.DiGraph) for graph in kwargs["t0_graphs"])
            assert all(nx.is_directed_acyclic_graph(graph) for graph in kwargs["t0_graphs"])
            self.t0_graph = kwargs["t0_graphs"][0]
            self.t0_ts = list(nx.topological_sort(self.t0_graph))
            self.gen_t0_graph = kwargs["t0_graphs"][1]
            self.gen_t0_ts = list(nx.topological_sort(self.gen_t0_graph))
            self.temporal = True

            assert set(self.t0_graph.nodes) == set(self.gen_t0_graph.nodes)
            assert all(isinstance(node, layer.InfGenProbabilityLayer)  for node in self.t0_graph.nodes)
        else:
            self.time_graph = nx.DiGraph()
            self.gen_time_graph = nx.DiGraph()
            self.t0_graph = None
            self.t0_ts = None
            self.gen_t0_graph = None
            self.gen_t0_ts = None
            self.temporal = False
        
        if "time_assoc" in kwargs:
            self.time_assoc = kwargs["time_assoc"]
        else:
            self.time_assoc = None

        if "gen_subgraph" in kwargs:
            #ensure that we actually have a valid subgraph of gen_graph
            gen_subgraph = kwargs["gen_subgraph"]
            assert isinstance(gen_subgraph, nx.DiGraph)
            assert nx.is_directed_acyclic_graph(gen_subgraph)
            assert set(gen_subgraph.nodes).issubset(set(self.gen_graph.nodes))
            assert set(gen_subgraph.edges).issubset(set(self.gen_graph.edges))
            self.gen_subgraph = gen_subgraph
        else:
            self.gen_subgraph = nx.DiGraph()

        self.gen_ts_subgraph = list(nx.topological_sort(self.gen_subgraph))
        
        #group the functions & parameters associated w/ inference and with generation into modules so they can be processed separately
        inf_list = []
        for node in self.graph.nodes:
            if isinstance(node, layer.InfGenProbabilityLayer) and isinstance(node.func,nn.Module):
                inf_list.append(node.func)
        gen_list = []
        for node in self.gen_graph.nodes:
            if isinstance(node, layer.InfGenProbabilityLayer) and isinstance(node.gen_func,nn.Module):
                gen_list.append(node.gen_func)
        
        inf_list_subgraph = []
        gen_list_subgraph = []
        for node in self.gen_subgraph.nodes:
            if isinstance(node.func, nn.Module):
                inf_list_subgraph.append(node.func)
            if isinstance(node.gen_func, nn.Module):
                gen_list_subgraph.append(node.gen_func)
        
        self.inf_group = nn.ModuleList(inf_list)
        self.gen_group = nn.ModuleList(gen_list)
        # self.inf_subgraph_group = nn.ModuleList(inf_list_subgraph)
        # self.gen_subgraph_group = nn.ModuleList(gen_list_subgraph)

        #InfGenNetworks that learn through reward must have a reward node that calculates rewards on a given trial.
        if "reward_node" in kwargs:
            self.reward_node = kwargs["reward_node"]
        else:
            self.reward_node = None
        
        self.reset()
        
    def forward(self, x: Tensor):
        x = x#[0] #pass through the inputs
        if not(type(x) is list):
            if torch.cuda.is_available():
                # y = torch.zeros(x.shape[0], self.n_classes, device = torch.cuda.current_device())
                y = F.one_hot(torch.randint(self.n_classes, [x.shape[0]], device = torch.cuda.current_device())).float()
            else:
                # y = torch.zeros(x.shape[0], self.n_classes)
                y = F.one_hot(torch.randint(self.n_classes, [x.shape[0]])).float()
            x = [x, y]
        else:
            x = [x[0], F.one_hot(x[1], num_classes = self.n_classes).float()]
        layer_len = len(self.graph.nodes)
        if self.temporal:
            self.temporal_cleanup()
        if self.t0 and self.temporal:
            ts = self.t0_ts
            graph = self.t0_graph
        else:
            ts = self.ts
            graph = self.graph
        for count, layer in enumerate(ts):
            input_list = []
            input_list += [parent.output for parent in graph.predecessors(layer)] #if at any intermediate layer, feed in the outputs of any parent layers
            # if not (len(self.time_graph.nodes)==0):
            #     input_list += [parent.prev_output for parent in self.time_graph.predecessors(layer)]
            if layer.input_layer: #child nodes receive the inputs to the network
                input_list += [*x]
            
            layer.forward(input_list, gen = False)
            
    def gen_forward(self, subgraph: DiGraph | None = None):
        #generate samples from the network
        if self.temporal:
            self.temporal_cleanup(gen = True)
        if not(subgraph is None):
            graph = subgraph
            ts = list(nx.topological_sort(graph))
        else:
            if self.t0 and self.temporal:
                ts = self.gen_t0_ts
                graph = self.gen_t0_graph
            else:
                ts = self.gen_ts
                graph = self.gen_graph
        for count, layer in enumerate(ts):
            input_list = []
            if list(graph.predecessors(layer)) == [] and (self.t0 or not(self.temporal)):
                input_list += [torch.tensor([float('nan')])]
            else:
                input_list += [parent.gen_output for parent in graph.predecessors(layer)]
                # if not (len(self.gen_time_graph.nodes)==0 or self.t0):
                #     input_list += [parent.prev_gen_output for parent in self.gen_time_graph.predecessors(layer)]
            
            layer.forward(input_list, gen = True)

    def mixed_forward(self, x: Tensor, mixing_constant: float, pyramid = False):
        #sets self.output for each layer as the stimulus-driven output
        if self.temporal:
            self.temporal_cleanup(mixed=True)
        self.forward(x)
        if self.t0 and self.temporal:
            ts = self.gen_t0_ts
            graph = self.gen_t0_graph
        else:
            ts = self.gen_ts
            graph = self.gen_graph
        for count, layer in enumerate(ts):
            input_list = []
            if list(graph.predecessors(layer)) == [] and (self.t0 or not(self.temporal)):
                input_list += [torch.tensor([float('nan')])]
            else:
                input_list += [parent.mixed_output for parent in graph.predecessors(layer)]
                # if not (len(time_graph.nodes)==0) and not(self.t0):
                #     input_list += [parent.prev_mixed_output for parent in time_graph.predecessors(layer)]
            if pyramid and count == 0:
                layer.mixed_forward(input_list, 0.)
            else:
                layer.mixed_forward(input_list, mixing_constant)

    def dynamic_mixed_forward(self, x: Tensor, T: int, mixing_constant: float, timescale: float, idxs: list[int], lesion_idxs: list[int] =[], apical_lesion_idxs: list[int] = [], mode = 'interp'):
        self.eval()
        with torch.no_grad():
            self.forward(x)
            record = []
            if not(type(x) is list):
                if torch.cuda.is_available():
                    # y = torch.zeros(x.shape[0], self.n_classes, device = torch.cuda.current_device())
                    y = F.one_hot(torch.randint(self.n_classes, [x.shape[0]], device = torch.cuda.current_device())).float()
                else:
                    # y = torch.zeros(x.shape[0], self.n_classes)
                    y = F.one_hot(torch.randint(self.n_classes, [x.shape[0]])).float()
                x = [x, y]
            else:
                x = [x[0], F.one_hot(x[1], num_classes = self.n_classes).float()]
            self.reset_dynamic_mixed_output()
            
            gen_ts = self.gen_ts
            gen_graph = self.gen_graph
            ts = self.ts
            graph = self.graph
            for idx in idxs:
                record.append(torch.zeros([T, *ts[idx].output.shape]))
            for tt in range(0, T):
                for count, layer in enumerate(gen_ts):
                    input_list_gen = []
                    input_list = []
                    if list(gen_graph.predecessors(layer)) == [] and (self.t0 or not(self.temporal)):
                        input_list_gen += [torch.tensor([float('nan')])]
                    else:
                        input_list_gen += [parent.dynamic_mixed_output_prev for parent in gen_graph.predecessors(layer)]
                    
                    input_list += [parent.dynamic_mixed_output_prev for parent in graph.predecessors(layer)] #if at any intermediate layer, feed in the outputs of any parent layers
                    if layer.input_layer: #child nodes receive the inputs to the network
                        input_list += [*x]
                    if count in apical_lesion_idxs:
                        layer.dynamic_mixed_forward(input_list, input_list_gen, mixing_constant, timescale, geometric_mean = False, apical_lesion = True, mode = mode)
                    else:
                        layer.dynamic_mixed_forward(input_list, input_list_gen, mixing_constant, timescale, geometric_mean = False, apical_lesion = False, mode = mode)
                    if count in lesion_idxs:
                        if torch.cuda.is_available():
                            layer.dynamic_mixed_output = torch.zeros(layer.dynamic_mixed_output.shape).to(torch.cuda.current_device())
                        else:
                            layer.dynamic_mixed_output = torch.zeros(layer.dynamic_mixed_output.shape)
                for count, idx in enumerate(idxs):  
                    record[count][tt,...] = self.ts[idx].dynamic_mixed_output
        self.train()
        return record

    def reset_dynamic_mixed_output(self):
        ts = self.gen_ts
        for layer in ts:
            layer.reset_dynamic_mixed_output()

    def log_prob(self, subgraph: DiGraph | None = None, mixed_output=False, regularizer: float = 0.):
        """
        Calculate the log probability of the network's inferred generative variables with respect to the inference probability model
        """
        if not(subgraph is None):
            graph = subgraph
            ts = list(nx.topological_sort(graph))
        else:
            if self.t0 and self.temporal:
                ts = self.t0_ts
                graph = self.t0_graph
            else:
                ts = self.ts
                graph = self.graph
        log_prob = 0
        for count, layer in enumerate(ts):
            input_list = []
            if layer.input_layer: #the first inference layers have no parameters to be evaluated, and p(s) from the environment is unknown
                input_list = []
            else:
                if mixed_output:
                    input_list += [parent.dynamic_mixed_output.detach() for parent in graph.predecessors(layer)]
                    # if not (len(self.time_graph.nodes)==0):
                    #     input_list += [parent.prev_mixed_output.detach() for parent in self.time_graph.predecessors(layer)]
                else:
                    input_list = [parent.gen_output.detach() for parent in graph.predecessors(layer)]
                    # if not (len(self.gen_time_graph.nodes)==0) and not(self.t0):
                    #     input_list += [parent.prev_gen_output.detach() for parent in self.gen_time_graph.predecessors(layer)]
                layer_log_prob = layer.log_prob(x = input_list, output = layer.gen_output.detach())
                log_prob += layer_log_prob#.sum(dim = list(range(1, layer_log_prob.dim())))
                if not(count == len(ts)-1):
                    l1_reg = regularizer * torch.abs(layer.predicted_activity_inf[0])#/layer.predicted_activity_inf[1]**2
                    log_prob -= torch.sum(l1_reg, dim = list(range(1,l1_reg.dim())))
        
        self.t0 = False
        return log_prob

    def gen_log_prob(self, subgraph: DiGraph | None = None, mixed_output = False, skip_last = False, differentiable = False, regularizer: float = 0.):
        """
        Calculate the log probability of the network's inferred latent variables with respect to the generative probability model
        """
        if not(subgraph is None):
            graph = self.gen_subgraph
            ts = list(nx.topological_sort(graph))
        else:
            if self.t0 and self.temporal:
                ts = self.gen_t0_ts
                graph = self.gen_t0_graph
            else:
                ts = self.gen_ts
                graph = self.gen_graph
                if skip_last:
                    ts = ts[1::]

        log_prob = 0
        for count, layer in enumerate(ts):
            if list(graph.predecessors(layer)) == [] and (self.t0 or not(self.temporal)):
                input_list = [torch.tensor([float('nan')])]
            else:
                if mixed_output:
                    input_list = [parent.dynamic_mixed_output.detach() for parent in graph.predecessors(layer)]
                    # if not (len(self.time_graph.nodes)==0):
                    #     input_list += [parent.prev_mixed_output.detach() for parent in self.time_graph.predecessors(layer)]
                else:
                    if not(differentiable): #if differentiable, allow gradients to flow through the network output
                        input_list = [parent.output.detach() for parent in graph.predecessors(layer)]
                    else:
                        input_list = [parent.output for parent in graph.predecessors(layer)]
                    # if not (len(self.time_graph.nodes)==0):
                    #     input_list += [parent.prev_output for parent in self.time_graph.predecessors(layer)]
            layer_log_prob = layer.log_prob(x = input_list, output = layer.output.detach(), gen = True)
            if layer_log_prob.dim() > 1:
                layer_log_prob = layer_log_prob.sum(dim = list(range(1, layer_log_prob.dim()))) #sum over non-batch dimensions
            log_prob += layer_log_prob
            if not(count == 0) and not(count == len(ts) -1 ):
                l1_reg = regularizer * torch.abs(layer.predicted_activity_gen[0])#/layer.predicted_activity_gen[1]**2
                log_prob -= torch.sum(l1_reg, dim = list(range(1,l1_reg.dim())))
        
        self.t0 = False
        return log_prob
    
    def reset(self):
        for layer in self.ts:
            layer.reset()
        if self.temporal:
            for layer in self.t0_ts:
                layer.reset()
        self.t0 = True

    def temporal_cleanup(self, mixed = False, gen = False):
        for l, l_prev in self.time_assoc:
            if (self.t0 and l.is_t0_layer) or (not(self.t0) and l.is_transition_layer):
                l_prev.set_output(l.get_output(mixed = mixed, gen = gen), mixed = mixed, gen = gen)