import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API
from flax.training import train_state

import optax

import numpy as np
import matplotlib.pyplot as plt

from typing import Sequence


class MLP(nn.Module):
  
  hidden_layer_nodes: Sequence[int]
  activation: str

  def setup(self):

    whitelist_activations = ['celu', 'elu', 'gelu', 'glu', 'log_sigmoid', 'log_softmax', 'relu', 'sigmoid', 'soft_sign', 'softmax', 'softplus', 'swish', 'PRelu']
    if self.activation not in whitelist_activations:
      raise ValueError(f'{self.activation} should be one of {whitelist_activations}')


    self.layers = [nn.Dense(n) for n in self.hidden_layer_nodes]
  
  def __call__(self, input):
    for layer in self.layer:
      x = layer(input)
      x = self.apply_activation(x)
      return x
  
  def apply_activation(self, input):
    if self.activation == 'celu': return nn.celu(input)
    elif self.activation == 'elu': return nn.elu(input)
    elif self.activation == 'gelu': return nn.gelu(input)
    elif self.activation == 'glu': return nn.glu(input)
    elif self.activation == 'log_sigmoid': return nn.log_sigmoid(input)
    elif self.activation == 'log_softmax': return nn.log_softmax(input)
    elif self.activation == 'relu': return nn.relu(input)
    elif self.activation == 'sigmoid': return nn.sigmoid(input)
    elif self.activation == 'soft_sign': return nn.soft_sign(input)
    elif self.activation == 'softmax': return nn.softmax(input)
    elif self.activation == 'softplus': return nn.softplus(input)
    elif self.activation == 'swish': return nn.swish(input)
    elif self.activation == 'PRelu': return nn.PRelu(input)
