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

    def _get_grads(self, grads_or_loss, params):
        # get grads if given is loss
        if isinstance(grads_or_loss, tensor.Tensor):
            grads = gradients(grads_or_loss, params)
        else:
            grads = grads_or_loss
        return grads
 

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
    """Gradient Descent Optimization

    Parameters
    ----------

    grads_or_loss: scalar tensor or list of gradients
        either the loss (scalar of Tensor type) to be differentied
        or the list of gradients already computed and possibly altered
        manually (such as clipping)

    params: list of parameters to update
        if grads_or_loss is al list then it should be ordered w.r.t. the
        given parameters

    learning_rate: constant or Tensor
        the learning rate use to update the parameters

    Attributes
    ----------

    updates: list of updates

    variables: list of variables

    """
 
    def __init__(self, grads_or_loss, params, learning_rate):
        grads = self._get_grads(grads_or_loss, params)

        if not numpy.isscalar(learning_rate) and not isinstance(
                learning_rate, tensor.Placeholder):
            learning_rate = learning_rate()

        updates = dict()
        for param, grad in zip(params, grads):
            updates[param] = param - learning_rate * grad
        self.updates = updates
        self.variables = []


class Adam(Optimizer):
    """Adaptive Gradient Based Optimization with renormalization

    Parameters
    ----------

    grads_or_loss: scalar tensor or list of gradients
        either the loss (scalar of Tensor type) to be differentied
        or the list of gradients already computed and possibly altered
        manually (such as clipping)

    params: list of parameters to update
        if grads_or_loss is al list then it should be ordered w.r.t. the
        given parameters

    learning_rate: constant or Tensor
        the learning rate use to update the parameters

    beta1: constant or Tensor
        the value of the exponential moving average of the average of the
        gradients through time (updates)

    beta2: constant or Tensor
        the value of the exponential moving average of the variance of the
        gradients through time

    Attributes
    ----------

    updates: list of updates

    variables: list of variables

    """
    def __init__(self, grads_or_loss, params, learning_rate, beta1=0.9,
                 beta2=0.999, epsilon=1e-6):

        grads = self._get_grads(grads_or_loss, params)
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
