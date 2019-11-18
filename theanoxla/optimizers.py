import numpy
from . import tensor
from .base import gradients


class PiecewiseConstantSchedule:
    def __init__(self, init, values):
        self.init = init
        self.values = values
    def __call__(self, step):
        return tensor.PiecewiseConstant(self.init, self.values, step)[0]

def SGD(params, grads, learning_rate):
    updates = dict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates


def Adam(grads_or_loss, params, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-6):

    # get grads if given is loss
    if isinstance(grads_or_loss, tensor.Tensor):
        grads = gradients(grads_or_loss, params)
    else:
        grads = grads_or_loss

    step = tensor.Variable(0., trainable=False, name='step')

    # get the learning rate
    if not numpy.isscalar(learning_rate) and not isinstance(learning_rate, tensor.Placeholder):
        learning_rate = learning_rate(step)

    updates = dict()
    for param, grad in zip(params, grads):
        m, update_m, _ = tensor.ExponentialMovingAverage(grad, beta1, step=step)
        v, update_v, _ = tensor.ExponentialMovingAverage(tensor.square(grad), beta2, step,
                                         init=numpy.ones(grad.shape))
        updates.update(update_m)
        updates.update(update_v)
        update = updates[m]/(tensor.sqrt(updates[v])+epsilon)
        updates[param] = param - learning_rate * update
    updates[step] = step + 1
    return updates
