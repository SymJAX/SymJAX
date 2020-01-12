import numpy
from . import tensor
from .base import gradients, function


class PiecewiseConstantSchedule:

    def __init__(self, init, values):
        self.init = init
        self.values = values
        self.step = tensor.Variable(0, trainable=False, name='step')
        self.value = tensor.PiecewiseConstant(self.init, self.values,
                                              self.step)[0]
        self._update = function(updates={self.step: self.step + 1})

    def update(self):
        self._update()

    def __call__(self):
        return self.value



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
    internal_variables = [step]
    # get the learning rate
    if not numpy.isscalar(learning_rate) and not isinstance(learning_rate, tensor.Placeholder):
        learning_rate = learning_rate()

    updates = dict()
    for param, grad in zip(params, grads):
        m, update_m, _ = tensor.ExponentialMovingAverage(grad, beta1, step=step)
        v, update_v, _ = tensor.ExponentialMovingAverage(tensor.square(grad), beta2, step,
                                         init=numpy.ones(grad.shape))
        internal_variables += [m, v]
        updates.update(update_m)
        updates.update(update_v)
        update = updates[m]/(tensor.sqrt(updates[v])+epsilon)
        updates[param] = param - learning_rate * update
    updates[step] = step + 1
    return updates, internal_variables
