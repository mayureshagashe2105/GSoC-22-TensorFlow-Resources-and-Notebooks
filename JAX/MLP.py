import jax.numpy as jnp
from jax import random
from jax import grad, jit, vmap
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import scipy


class MLP:

  __KEY = random.PRNGKey(21)

  def __init__(self, units: list, activations: list, learning_rate=0.001):
    
    self.layers = len(units)
    self.units = jnp.array(units)
    self.activations = activations
    self.lr = learning_rate

    self.weights = [random.uniform(MLP.__KEY, shape=(self.units[i + 1], self.units[i]), 
                                   dtype=jnp.float32, minval=-1.0) for i in range(self.layers - 1)]
    self.biases = [jnp.zeros((i,), dtype=jnp.float32) for i in self.units[1:]]

    self.batch_size = None
    self.loss = 0
    self.logs = {"loss": [], "accuracy": []}
  

  def __forward_prop(self, input_jnp):

    self.intermediates = [input_jnp,]

    for layer in range(self.layers - 1):
      inter_state = MLP.matmul_fn(self.intermediates[-1], self.weights[layer].T)
      inter_state += self.biases[layer]
      inter_state = self.__apply_activation_fn(inter_state, layer)
      self.intermediates.append(inter_state)

    
    self.batch_output = self.intermediates[-1]
    self.intermediates = self.intermediates[:-1]
  

  def __categorical_cross_entropy(self, preds, actual_outputs):
    one_hot_encoded_matrix = np.zeros((self.batch_size, self.units[-1])) + 0.01
    one_hot_encoded_matrix[[i for i in range(self.batch_size)], actual_outputs] = 0.99

    self.batch_ohe_matrix = jnp.array(one_hot_encoded_matrix, dtype=jnp.float32)

    self.error = np.sum((one_hot_encoded_matrix * jnp.log(preds)), axis=0)
    self.error =  -1 / preds.shape[-1] * np.sum(self.error)



    self.loss += self.error
  

  def __back_prop(self):

    cum_common_grad = self.batch_output - self.batch_ohe_matrix

    for i in range(self.layers - 2, -1, -1):
      grads = jnp.expand_dims(cum_common_grad, -1)
      prevs = jnp.expand_dims(self.intermediates[i], 1)
      cum_grads = MLP.matmul_fn(grads, prevs)
      cum_grads = jnp.mean(cum_grads, axis=0)
      self.weights[i] -= (self.lr * cum_grads)
      self.biases[i] -= (self.lr * np.mean(cum_common_grad, axis=0))
      cum_common_grad = MLP.matmul_fn(cum_common_grad, self.weights[i]) * (self.intermediates[i] * (1 - self.intermediates[i]))
    
  
  def fit(self, X, y, batch_size, epochs):
    self.batch_size = batch_size
    num_batches = X.shape[0] // batch_size
    for epoch in range(epochs):
      score = 0
      for idx in range(num_batches):
        batch_inputs = X[idx * self.batch_size: (idx + 1) * self.batch_size, :]
        actual_outputs = y[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        self.__forward_prop(batch_inputs)
        self.__categorical_cross_entropy(self.batch_output, actual_outputs)
        self.__back_prop()
        score += self.__score(actual_outputs)

        print(f"\r{idx + 1} / {num_batches}", end=" ")

      acc = score / X.shape[0]
      self.logs["loss"].append(self.loss)
      self.logs["accuracy"].append(acc)
      print(f"\rEpoch {epoch + 1} / {epochs}\nloss: {self.loss} accuracy: {acc}")
      self.loss = 0
  
  def __score(self, actual_outputs):
    preds = jnp.argmax(self.batch_output, axis=1)
    preds = jnp.sum((preds == actual_outputs).astype(jnp.int32))
    return preds

  def __apply_activation_fn(self, inter_state, layer):
    
    if self.activations[layer] == 'softmax':
      return(MLP.softmax_activation_fn(inter_state))
    
    elif self.activations[layer] == 'sigmoid':
      return(MLP.sigmoid_activation_fn(inter_state))

      

  @staticmethod  
  @jit
  def sigmoid_activation_fn(x):
    return 1.0 / (1.0 + jnp.exp(-x))
  

  @staticmethod  
  @jit
  def relu_activation_fn(x):
    return jnp.maximum(0.001, x)
  
  @staticmethod
  def softmax_activation_fn(x):
    return scipy.special.softmax(x)
  

  @staticmethod
  @jit
  def matmul_fn(x, y):
    return jnp.matmul(x, y)

