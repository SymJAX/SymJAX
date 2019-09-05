from ..tensor.base import Op
import jax.numpy as np


def _relu(x): return np.maximum(x, 0)
def softplus(x): return np.logaddexp(x, 0)
def soft_sign(x): return x / (np.abs(x) + 1)
def sigmoid(x): return expit(x)
def swish(x): return x * sigmoid(x)
def log_sigmoid(x): return -softplus(-x)

def elu(x, alpha=1.0):
  return np.where(x > 0, x, alpha * np.expm1(x))

def leaky_relu(x, negative_slope=1e-2):
  return np.where(x >= 0, x, negative_slope * x)

def hard_tanh(x):
  return np.where(x > 1, 1, np.where(x < -1, -1, x))

def celu(x, alpha=1.0):
  """Continuously-differentiable exponential linear unit activation"""
  return np.where(x > 0, x, alpha * np.expm1(x / alpha))

def selu(x):
  """Scaled exponential linear unit activation"""
  alpha = 1.6732632423543772848170429916717
  scale = 1.0507009873554804934193349852946
  return scale * leaky_relu(x, alpha)

def gelu(x):
  """Gaussian error linear unit activation"""
  return x * (lax.erf(x / np.sqrt(2)) + 1) / 2

def glu(x, axis=-1):
  """Gated linear unit activation"""
  size = x.shape[axis]
  assert size % 2 == 0, "axis size must be divisible by 2"
  return x[..., :size] * sigmoid(x[..., size:])

# other functions

def log_softmax(x, axis=-1):
  shifted = x - x.max(axis, keepdims=True)
  return shifted - np.log(np.sum(np.exp(shifted), axis, keepdims=True))

def softmax(x, axis=-1):
  unnormalized = np.exp(x - x.max(axis, keepdims=True))
  return unnormalized / unnormalized.sum(axis, keepdims=True)


def _relu(x):
    return np.maximum(x, 0.)

def _leaky_relu(x, alpha=0.01):
    return np.maximum(x, alpha)

leaky_relu = Op(_leaky_relu, name='leaky_relu')
relu = Op(_relu, name='relu')






