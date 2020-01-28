import numpy
from . import tensor
from .base import gradients, function


class Optimizer:

    def reset(self):
        for var in self.variables:
            var.reset()

    def update(self):
        if '_update' in self.__dict__:
            self._update()
        else:
            self._update = function(updates=self.updates)
            self._update()


# class PiecewiseConstant(Optimizer):
#
#    def __init__(self, init, values):
#        self.init = init
#        self.values = values
#        self.step = tensor.Variable(0, trainable=False, name='step')
#        self.value = tensor.PiecewiseConstant(self.init, self.values,
#                                              self.step)[0]
#        self.updates = {self.step: self.step + 1}
#        self.variables = [self.step]
#
#    def __call__(self):
#        return self.value


class SGD(Optimizer):
    def __init__(self, grads_or_loss, params, learning_rate):
        # get grads if given is loss
        if isinstance(grads_or_loss, tensor.Tensor):
            grads = gradients(grads_or_loss, params)
        else:
            grads = grads_or_loss

        updates = dict()
        for param, grad in zip(params, grads):
            updates[param] = param - learning_rate * grad
        self.updates = updates
        self.variables = []


class Adam(Optimizer):
    def __init__(self, grads_or_loss, params, learning_rate, beta1=0.9,
                 beta2=0.999, epsilon=1e-6):

        # get grads if given is loss
        if isinstance(grads_or_loss, tensor.Tensor):
            grads = gradients(grads_or_loss, params)
        else:
            grads = grads_or_loss
        step = tensor.Variable([[0.]], trainable=False, name='step')
        variables = [step]
        # get the learning rate
        if not numpy.isscalar(learning_rate) and not isinstance(
                learning_rate, tensor.Placeholder):
            learning_rate = learning_rate()

        updates = dict()
        for param, grad in zip(params, grads):
            m, update_m, _ = tensor.ExponentialMovingAverage(grad, beta1,
                                                             step=step)
            v, update_v, _ = tensor.ExponentialMovingAverage(
                tensor.square(grad), beta2, step, init=numpy.ones(grad.shape))
            variables += [m, v]
            updates.update(update_m)
            updates.update(update_v)
            update = updates[m] / (tensor.sqrt(updates[v]) + epsilon)
            updates[param] = param - learning_rate * update
        updates[step] = step + 1
        self.updates = updates
        self.variables = variables
