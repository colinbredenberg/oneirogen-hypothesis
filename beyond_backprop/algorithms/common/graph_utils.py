from __future__ import annotations
import torch
import os
import networkx as nx


def sequential_graph(node_list: list):
    """
    From a basic ordered list of objects, construct a directed graph with the identical ordering
    Particularly useful for constructing a feedforward layered network architecture

    Inputs
    ------
    node_list:
        list of nodes to connect in order
    
    Outputs
    -------
    graph:
        nx.DiGraph constructed from the list of nodes
    """
    assert isinstance(node_list, list) #make sure that the input is a list
    assert len(set(node_list)) == len(node_list) #make sure that there are only unique objects in the node_list

    graph = nx.DiGraph()
    for (counter, node) in enumerate(node_list):
        if counter == 0:
            graph.add_node(node)
        else:
            graph.add_edge(node_list[counter -1], node_list[counter])
    
    return graph