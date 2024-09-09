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

def multi_stream_graph(stream_list: list[list], reverse = False):
    """
    Constructs N sequential graphs from from a list of individual sequences. The last node in each stream must be the same, and all streams are connected at this node
    The graph is structured as stream_1 -> final_node <- stream_2 (the same pattern holds for N>2)
    If reverse is True, the graph is structured as stream_1 <- final_node -> stream_2 (the same pattern holds for N > 2)

    Inputs
    ------
    stream_list:
        list of lists containing nodes to connect in order. The final node in each node list must be common across all node lists
    reverse:
        Boolean. If true, the graph is constructed then the edge directions are all reversed
    Outputs
    -------
    graph:
        nx.DiGraph constructed from the two streams of nodes
    """
    assert all([isinstance(stream, list) for stream in stream_list])
    assert all([stream[-1] == stream_list[0][-1] for stream in stream_list]) #require the final node to be shared by all streams

    graph = nx.DiGraph()
    #connect the first stream
    for stream in stream_list:
        for (counter, node) in enumerate(stream):
            if counter == 0:
                graph.add_node(node)
            else:
                graph.add_edge(stream[counter-1], stream[counter])
    
    if reverse:
        graph = graph.reverse()
    
    return graph