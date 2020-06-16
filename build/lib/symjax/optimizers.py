import numpy

import symjax
from . import tensor
from .base import gradients, function


class Optimizer:

    def reset(self):
        if hasattr(self, 'variables'):
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

    def __init__(self, grads_or_loss, learning_rate, params=None):

        if params is None:
            params = [v for k, v in get_graph().variables.items() if
                      v.trainable]
        elif type(params) is not list and type(params) is not tuple:
            params = [params]

        grads = self._get_grads(grads_or_loss, params)

        if not numpy.isscalar(learning_rate) and not isinstance(
                learning_rate, tensor.Placeholder):
            learning_rate = learning_rate()

        updates = dict()
        for param, grad in zip(params, grads):
            updates[param] = param - learning_rate * grad

        self.updates = updates


#        if get_graph() is not None:
#            get_graph().updates.update(updates)


class NesterovMomentum(Optimizer):
    """Nesterov momentum Optimization

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

    def __init__(self, grads_or_loss, learning_rate, momentum, params=None):

        if params is None:
            params = [v for k, v in get_graph().variables.items() if
                      v.trainable]

        grads = self._get_grads(grads_or_loss, params)

        if not numpy.isscalar(learning_rate) and not isinstance(
                learning_rate, tensor.Placeholder):
            learning_rate = learning_rate()

        updates = dict()
        variables = []
        for param, grad in zip(params, grads):
            velocity = tensor.Variable(
                numpy.zeros(param.shape, dtype=param.dtype),
                trainable=False)
            variables.append(velocity)
            update = param - learning_rate * grad
            x = momentum * velocity + update - param
            updates[velocity] = x
            updates[param] = momentum * x + update

        self.variables = variables
        self.updates = updates
        if get_graph() is not None:
            get_graph().updates.update(updates)


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

    def __init__(self, grads_or_loss, learning_rate, beta1=0.9,
                 beta2=0.999, epsilon=1e-6, params=None):

        if params is None:
            params = symjax.get_variables("*", trainable=True)

        grads = self._get_grads(grads_or_loss, params)
        step = tensor.Variable(tensor.zeros(1, dtype='float32'),
                               trainable=False, name='step')
        variables = [step]
        # get the learning rate
        if callable(learning_rate):
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

        self.variables = variables
        self.updates = updates
#        if get_graph() is not None:
#            get_graph().updates.update(updates)
