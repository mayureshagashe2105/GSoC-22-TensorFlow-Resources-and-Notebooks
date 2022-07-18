class MaxPool2D:

  def __init__(self, mask_size, stride, input_shape):
    self.mask_size = mask_size
    self.stride = stride
    self.n_channles = input_shape[-1]
    self.input_shape = input_shape

    self.out_y = (input_shape[0] - self.mask_size[0]) // self.stride + 1
    self.out_x = (input_shape[1] - self.mask_size[1]) // self.stride + 1
  
  def forward_prop(self, batch_inputs):
    self.batch_size = batch_inputs.shape[0]
    windows = jnp.array(np.lib.stride_tricks.as_strided(batch_inputs, (self.batch_size, self.out_y, self.out_x, self.n_channles, *self.mask_size),
                                              (batch_inputs.strides[0], batch_inputs.strides[1] * self.stride, batch_inputs.strides[2] * self.stride, batch_inputs.strides[3], 
                                              batch_inputs.strides[1], batch_inputs.strides[2])
                                              ))
    
    self.pooled_out = jnp.max(windows, axis=(-2, -1))

    max_masks = self.pooled_out.repeat(self.mask_size[0], axis=1).repeat(self.mask_size[1], axis=2)
    batched_window_input = batch_input[:, :self.out_y * self.stride, :self.out_x * self.stride, :]
    self.mask = jnp.equal(batched_window_input, max_masks).astype(jnp.int32)
  

  def back_prop(self, grad_from_next_layer):
    grad_from_next_layer = grad_from_next_layer.repeat(self.mask_size[0], axis=1).repeat(self.mask_size[1], axis=2)
    grad_from_next_layer = MaxPool2D.multiply_fn(grad_from_next_layer, self.mask)
    padding = jnp.zeros((self.batch_size, *self.input_self))
    padding[:, grad_from_next_layer[1], grad_from_next_layer[3], :] = grad_from_next_layer
    self.grad = padding
  


  @staticmethod
  @jit
  def multiply_fn(x, y):
    return jnp.multiply(x, y)

    
