import beyond_backprop.algorithms.common.layer as layer
import networkx as nx
import torch


def weight_mirror_update(network, params):
    for i in range(params.wm_batch_size):
        for counter, node in enumerate(network.ts):
            if isinstance(node,layer.FuncLayerWrapper):
                noise_input = 0.1* torch.randn(params.wm_batch_size,node.module.weight.size(dim=1))
                noise_output = node.forward([noise_input])
                node.module.weight_backward.data += 0.5 * noise_output.t().mm(noise_input)
                #node.module.bias_backward.data += noise_output.sum(0).squeeze(0)

    for counter, node in enumerate(network.ts):
        if isinstance(node, layer.FuncLayerWrapper):
            x = torch.randn(node.module.weight_backward.size(dim=1),params.wm_batch_size)
            y = node.module.weight_backward.mm(x)
            y_std = torch.mean(torch.std(y,dim=0))
            node.module.weight_backward.data = 0.5*node.module.weight_backward.data/y_std
            #node.module.bias_backward.data = 0.5 * node.module.bias_backward.data / y_std

def kolen_pollack_update(network, params):
    for counter, node in enumerate(network.ts):
        if isinstance(node,layer.FuncLayerWrapper):
            node.module.weight.data = params.kp_weight_decay * node.module.weight.data - (params.kp_learning_rate )*  node.module.weight.grad.data
            node.module.bias.data = node.module.bias.data - (params.kp_learning_rate)* node.module.bias.grad.data
            node.module.weight_backward.data = params.kp_learning_rate * node.module.weight_backward.data - (params.kp_learning_rate)*  node.module.weight.grad.data