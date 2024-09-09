import torch
from torch import nn, Tensor
from beyond_backprop.algorithms.common.graph_utils import sequential_graph, multi_stream_graph
import torch.nn.functional as F

import networkx as nx
from . import layer
from dataclasses import dataclass

#%% Broad Network definitions (not corresponding to any particular architecture)

class LayeredNetwork(nn.Module):
    """
    Define a network that is explicitly composed of a directed acyclic graph of layers, such that the forward operation corresponds to running through the graph

    Attributes
    ---------
    graph: nx.Digraph
        DAG specifying the order of computation for the forward method
    ts: list
        List specifying an iterable ordering of nodes such that the required inputs from parents are always available for a child node
    
    Methods
    -------
    forward:
        Iterate through self.ts to calculate the activations at each layer in response to inputs
    """
    def __init__(self, **kwargs):
        super().__init__()
        if "graph" in kwargs:
            assert isinstance(kwargs["graph"], nx.DiGraph) #verify that the graph fed in for initialization is a valid directed graph
            assert nx.is_directed_acyclic_graph(kwargs["graph"]) #verify that the graph is a valid directed acyclic graph
            self.graph = kwargs["graph"]

            for node in self.graph.nodes: #verify that each node in the graph is a valid instance of the Layer class
                assert isinstance(node, layer.Layer) or isinstance(node, layer.ProbabilityLayer)
        else:
            self.graph = nx.DiGraph()

        if "time_graph" in kwargs:
            assert isinstance(kwargs["temp_graph"], nx.DiGraph)
            assert nx.is_directed_acyclic_graph(kwargs["graph"])
            self.time_graph = kwargs["temp_graph"]

            for node in self.time_graph.nodes:
                assert isinstance(node, layer.Layer) or isinstance(node, layer.ProbabilityLayer)
        else:
            self.time_graph = nx.DiGraph()

        #the topological sorter provides an iterable list such that you process the graph in an order from parents -> children without conflicts
        self.ts = list(nx.topological_sort(self.graph))

    def forward(self, x: Tensor) -> Tensor:
        #y = x[1] #pass through the targets
        #x = x[0] #pass through the inputs
        layer_len = len(self.graph.nodes)
        input_list = []
        #iterate through the topological sort to compute the activations at each node in self.graph
        for count, layer in enumerate(self.ts):
            if count == 0:
                input_list += [x]
            input_list += [parent.output for parent in self.graph.predecessors(layer)] #if at any intermediate layer, feed in the outputs of any parent layers
            if not (len(self.time_graph.nodes)==0):
                input_list += [parent.prev_output for parent in self.time_graph.predecessors(layer)]
            
            layer.forward(input_list)
        
        for layer in self.ts:
            layer.cleanup()
        return layer.output #return the output of the last layer

    def reset(self):
        for layer in self.ts:
            layer.reset()
