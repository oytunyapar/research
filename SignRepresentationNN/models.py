import torch.nn as nn
from torch.nn.modules.module import Module


class SigmaPiModel(Module):
    def __init__(self, dimension):
        super(SigmaPiModel, self).__init__()
        self.linear1 = nn.Linear(2 ** dimension, 1, bias=False)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return x
