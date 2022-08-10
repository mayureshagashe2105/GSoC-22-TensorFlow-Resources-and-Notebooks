import numpy as np
import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap
from tensorflow.image import extract_patches


class ConvLayer:

  __KEY = random.PRNGKey(21)

  def __init__(self, n_filters, filter_size, stride, input_shape):

    self.filter_size = filter_size
    self.stride = stride
    self.n_filters = n_filters
    self.channels = input_shape[-1]

    self.filters = random.normal(ConvLayer.__KEY, shape=(filter_size[0], filter_size[1], self.n_filters), dtype=jnp.float64)
    
    self.output_dim_x = (input.shape[0] - self.filter_size[0]) // self.stride[0] + 1
    self.output_dim_y = (input.shape[1] - self.filter_size[1]) // self.stride[1] + 1

  
  def __forward_prop(self, input_batch):
    patches = jnp.array(extract_patches(input_batch, sizes=[1, self.filter_size[0], self.filter_size[1], 1],
                              strides=[1, self.stride[0], self.stride[1], 1],
                              rates=[1, 1, 1, 1], padding="VALID").numpy(), dtype=jnp.float64
                              ).reshape(input_batch.shape[0], self.output_dim_x, self.output_dim_y, 
                                        self.channels, self.filter_size[0], self.filter_size[1])

    convolved_output = jnp.einsum("ijklmn, mno -> ijko", patches, self.filters)


    self.conv_grad = patches
    self.batch_convolved_output = convolved_output

  def __back_prop(self, grad_from_next_layer, lr):
    dl_dw = jnp.einsum("ijklmn, ijklmn -> jklmn", self.conv_grad, grad_from_next_layer)
    self.filters -= (lr * dl_dw)
    
