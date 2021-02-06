import torch


def create_dqn_agent(layer_sizes):
    model_layers = []
    for layer_iterator in range(len(layer_sizes) - 1):
        model_layers.append(torch.nn.Linear(layer_sizes[layer_iterator], layer_sizes[layer_iterator + 1]))
        model_layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*model_layers)
