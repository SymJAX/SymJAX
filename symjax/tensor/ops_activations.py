from .. import tensor as T

def relu(x): return T.maximum(x, 0)

def softplus(x): return T.logaddexp(x, 0)

def soft_sign(x): return x / (T.abs(x) + 1)

def sigmoid(x): return 1 / (1 + T.exp(-x))

def swish(x): return x * sigmoid(x)

def log_sigmoid(x): return -softplus(-x)

def elu(x, alpha=1.0):
    safe_x = T.where(x > 0, 0., x)
    return T.where(x > 0, x, alpha * T.expm1(safe_x))

def leaky_relu(x, leakiness=0.01):
    return T.where(x >= 0, x, leakiness * x)

def softmax(x, axis=-1):
    maxv = T.stop_gradient(T.max(x, axis=axis, keepdims=True))
    expv = T.exp(x-maxv)
    return expv / expv.sum(axis=axis, keepdims=True)
