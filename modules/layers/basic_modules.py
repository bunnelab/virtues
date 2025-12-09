import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any, List



def build_activation(activation_fn: str, activation_params: Dict[str, Dict[str, Any]] = {}) -> nn.Module:
    """
    Build activation function from string.
    Supported activations: 'relu', 'leaky_relu', 'elu', 'gelu', 'sigmoid', 'tanh', 'silu', 'selu', 'softplus', 'softsign', 'identity'.
    If the activation function is not recognized, defaults to nn.Identity().
    Args:
        activation_fn (str): Name of the activation function.
        activation_params (Dict[str, Dict[str, Any]]): Dictionary of parameters for the activation function.
            Example: {'relu': {'inplace': True}, 'leaky_relu': {'negative_slope': 0.01}}
    Returns:
        nn.Module: Activation function module.
    """

    # for activation functions which have __init__, check if parameters exist in activation_kwargs
    return {
        "relu": nn.ReLU(**activation_params.get("relu", {})),
        "leaky_relu": nn.LeakyReLU(**activation_params.get("leaky_relu", {})),
        "elu": nn.ELU(**activation_params.get("elu", {})),
        "gelu": nn.GELU(**activation_params.get("gelu", {})),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "silu": nn.SiLU(),
        "selu": nn.SELU(),
        "softplus": nn.Softplus(**activation_params.get("softplus", {})),
        "softsign": nn.Softsign(),
        "identity": nn.Identity(),
    }.get(activation_fn.lower(), nn.Identity())  


def build_feedforward(in_dim: int, out_dim: int, hidden_dims: Union[int, Tuple[int, ...]], 
                      activation_fn: Union[str, List[str]] = "relu", activation_params: Dict[str, Dict[str, Any]] = {},
                      use_dropout: bool = True, dropout_prob: Union[float, List[float]] = 0.5) -> nn.Sequential:
    """
    Build a feedforward neural network.
    Args:
        in_dim (int): Input dimension.
        out_dim (int): Output dimension.
        hidden_dims (Union[int, Tuple[int, ...]]): Hidden layer dimensions. If int, a single hidden layer is created.
            If tuple, multiple hidden layers are created with the specified dimensions.
        activation_fn (str | List[str]): Activation function to use between layers. Default is 'relu'. Can be a list to specify different activations for each layer.
        activation_params (Dict[str, Dict[str, Any]]): Dictionary of parameters for the activation function.
            Example: {'relu': {'inplace': True}, 'leaky_relu': {'negative_slope': 0.01}}
    Returns:
        nn.Sequential: Feedforward neural network module.
    """
    
    
    if isinstance(hidden_dims, int):
        hidden_dims = (hidden_dims,)

    if isinstance(activation_fn, str):
        activation_fn = [activation_fn] * (len(hidden_dims))
    
    if len(activation_fn) != len(hidden_dims):
        raise ValueError("Length of activation_fn list must be equal to number of layers (hidden).")

    if use_dropout:
        if isinstance(dropout_prob, float):
            dropout_prob = [dropout_prob] * (len(hidden_dims))
        if len(dropout_prob) != len(hidden_dims):
            raise ValueError("Length of dropout_prob list must be equal to number of layers (hidden).")
        
    layers = []
    prev_dim = in_dim
    for i, hidden_dim in enumerate(hidden_dims):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(build_activation(activation_fn[i], activation_params))
        if use_dropout:
            layers.append(nn.Dropout(dropout_prob[i]))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, out_dim))

    return nn.Sequential(*layers)
    