import torch
from torch import nn
import numpy as np
from itertools import zip_longest

from typing import List, Union


def create_mlp(
        layer_sizes: Union[List[int], int], activations: Union[type, List[type]],
        in_size: int, out_size: int = None, dropout: float = 0.0,
        last_lin: bool = False, bias: bool = True, out_act=None,
):
    """
    Quickly build an MLP module.

    :param layer_sizes: Array with the size of each layer
    :param activations: Activation or Array of activations
    :param in_size: Number of expected inputs
    :param out_size: Number of expected outputs
    :param dropout: Dropout p value
    :param last_lin: Whether the last layer should have linear activation instead of the one provided
    :param bias: Whether to use bias in all the layers.
    :return:
    """
    layers = []
    last_size = in_size
    if isinstance(layer_sizes, int):
        layer_sizes = [layer_sizes]

    def init_weights(m, slope=0.25):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight, np.sqrt((2 / (1 + slope ** 2))))

    if activations is not None and issubclass(activations, nn.Module):
        # if the provided activation is a constuctor, initialize one for each layer
        activations = [activations() for i in range(len(layer_sizes))]
    else:
        activations = [activations for i in range(len(layer_sizes))]

    if last_lin and activations is not None:
        activations[-1] = None
    for i, (ls, act) in enumerate(zip_longest(layer_sizes, activations)):
        if ls is not None:
            layers.append(nn.Linear(last_size, ls, bias=bias))
            if act is not None:
                layers[-1].apply(init_weights)
            last_size = ls
        if act is not None:
            layers.append(act)
        if dropout > 0 and ls is not None:
            layers.append(nn.Dropout(dropout))
    if out_size is not None:
        layers.append(nn.Linear(last_size, out_size, bias=bias))
    if out_act is not None:
        layers.append(out_act())
    mlp = nn.Sequential(*layers)
    return mlp
