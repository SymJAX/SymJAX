from ..tensor.base import Op
import jax.numpy as np


def _sigmoid(x):
    return 1./(1.+np.exp(x))

def _silu(x):
    return x * _sigmoid(x)

def _hard_tanh(x):
  return np.where(x > 1, 1, np.where(x < -1, -1, x))

def _log_softmax(x, axis=-1):
  shifted = x - x.max(axis, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis, keepdims=True))

def _softmax(x, axis=-1):
  unnormalized = np.exp(x - x.max(axis, keepdims=True))
  return unnormalized / unnormalized.sum(axis, keepdims=True)

def _relu(x):
    return np.maximum(x, 0.)

def _leaky_relu(x, alpha=0.01):
    return np.maximum(x, x*alpha)


leaky_relu = Op(_leaky_relu, name='leaky_relu',
                docstring="leaky-relu activation function")

relu = Op(_relu, name='relu',
          docstring="relu activation function")

sigmoid = Op(_sigmoid, name='sigmoid',
             docstring="sigmoid activation function")

hard_tanh = Op(_hard_tanh, name='hard_tanh',
               docstring="hard-tanh activation function")

silu = Op(_silu, name='silu',
          docstring="sigmoid gated linear unit activation function")

softmax = Op(_softmax, name='softmax',
             docstring="softmax activation function")
