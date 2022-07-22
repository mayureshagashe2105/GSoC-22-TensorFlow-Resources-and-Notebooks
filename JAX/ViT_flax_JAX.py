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


    
class PatchEncoder(nn.Module):
  num_patches: int
  projection_dims: int

  def setup(self):
    self.projection = nn.Dense(self.projection_dims)
    self.positional_encodings = nn.Embed(self.num_patches, self.projection_dims)
  
  def __call__(self, patch):
    positions = jnp.arange(0, self.num_patches)
    encode = self.projection(patch) + self.position_encodings(positions)
    return encode

class ExtractPatches(nn.Module):
  patch_size: Sequence[int]
  stride: int


  def __call__(self, inputs):
    batch_size = inputs.shape[0]
    patches = tf.image.extract_patches(inputs, sizes=[1, self.patch_size[0], self.patch_size[1], 1],
                                       strides=[1, self.stride, self.stride, 1],
                                       rates=[1, 1, 1, 1],
                                       padding="VALID"
                                       )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])
    return patches
