import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn  # nn notation also used in PyTorch and in Flax's older API
from flax.training import train_state
from flax.linen.transforms import jit

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

  

class VisionTransformer(nn.Module):
  patch_size: Sequence[int]
  stride: int
  image_size: Sequence[int]
  activation: str
  projection_dims: int
  num_heads: int
  transformer_layers: int
  mlp_head_units: Sequence[int]
  batch_size: int
  num_classes: int
  learning_rate: float



  def setup(self):
    self.transformer_units = [self.projection_dims * 2, self.projection_dims]
    self.num_patches = ((self.image_size[0] - self.patch_size[0]) // self.stride + 1) * ((self.image_size[1] - self.patch_size[1]) // self.stride + 1)
    
    self.norm = nn.LayerNorm(epsilon=1e-6)
    self.multi_head_attention = nn.MultiHeadDotProductAttention(num_heads=self.num_heads, qkv_features=self.projection_dims)
    self.dropout10 = nn.Dropout(0.1)
    self.dropout50 = nn.Dropout(0.5)
    self.logits = nn.Dense(self.num_classes)

    patches_init = ExtractPatches(self.patch_size, self.stride)
    encode_init = PatchEncoder(self.num_patches, self.projection_dims)
    mlp_init = MLP(self.transformer_units[::-1], self.activation)
    mlp2_init = MLP(self.mlp_head_units[::-1], self.activation)

    self.patches = patches_init
    self.encode = encode_init
    self.mlp = mlp_init
    self.mlp2 = mlp2_init
  
  
  @nn.compact
  def __call__(self, inputs):
    image_patches = self.patches(inputs)
    encoded_image_patches = self.encode(image_patches)

    for _ in range(self.transformer_layers):
      x1 = self.norm(encoded_image_patches)
      attention_output = self.multi_head_attention(x1, x1)
      x2 = attention_output + encoded_image_patches #VisionTransformer.layer_add(attention_output, encoded_image_patches)
      x3 = self.norm(x2)

      x3 = self.mlp(x3)
      # x3 = self.dropout10(x3, deterministic=not True)
      # print(x3.shape)
      encoded_image_patches = x3 + x2 #VisionTransformer.layer_add(x3, x2)
    
    repr = self.norm(encoded_image_patches)
    repr = repr.reshape(-1,)
    # repr = self.dropout50(repr, deterministic=not True)
    repr = self.mlp2(repr)
    # print(repr.shape)
    # repr = self.dropout(repr, deterministic=not True)
    logit_nodes = self.logits(repr)
    logit_nodes = nn.softmax(logit_nodes)

    return logit_nodes



  @staticmethod
  @jit
  def layer_add(x, y):
    return jnp.add(x, y)

  
  
rng = jax.random.PRNGKey(0) # PRNG Key
x = jnp.ones(shape=(256, 32, 32, 3)) # Dummy Input
params = model.init(rng, x) # Initialize the parameters
jax.tree_map(lambda x: x.shape, params) # Check the parameters


def init_train_state(
    model, random_key, shape, learning_rate
):
    # Initialize the Model
    variables = model.init(random_key, jnp.ones(shape))
    # Create the optimizer
    optimizer = optax.adam(learning_rate)
    # Create a State
    return train_state.TrainState.create(
        apply_fn = model.apply,
        tx=optimizer,
        params=variables['params']
    )

state = init_train_state(
    model, rng, (64, 32, 32, 3), 0.001
)


def compute_metrics(*, logits, gt_labels):
    one_hot_gt_labels = jax.nn.one_hot(gt_labels, num_classes=100)
    loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
    accuracy = jnp.mean(jnp.argmax(logits, -1) == gt_labels)

    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics
  

  @jax.jit
def train_step(
    state, image:jnp.ndarray, label:jnp.ndarray
):


    def loss_fn(params):
        logits = model.apply({'params': params}, image)
        one_hot_gt_labels = jax.nn.one_hot(label, num_classes=100)
        loss = -jnp.mean(jnp.sum(one_hot_gt_labels * logits, axis=-1))
        return loss, logits


    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = gradient_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, gt_labels=label)
    return state, metrics


@jax.jit
def eval_step(state, batch):
    image, label = batch
    logits = state.apply_fn({'params': state.params}, image)
    return compute_metrics(logits=logits, labels=label)

  
def train_one_epoch(state, dataloader, epoch):
  """Train for 1 epoch on the training set."""
  batch_metrics = []
  for cnt, (imgs, labels) in enumerate(dataloader):
      print(cnt, end=" ")
      state, metrics = train_step(state, imgs, labels)
      batch_metrics.append(metrics)
      print("\r", end = " ")
  
  # Aggregate the metrics
  batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]
  }
  return state, epoch_metrics_np


seed = 21  # needless to say these should be in a config or defined like flags
learning_rate = 0.001
momentum = 0.9
num_epochs = 2
batch_size = 64

train_state = state

for epoch in range(1, num_epochs + 1):
    train_state, train_metrics = train_one_epoch(train_state, gen, epoch)
    print(f"Train epoch: {epoch}, loss: {train_metrics['loss']}, accuracy: {train_metrics['accuracy'] * 100}")
