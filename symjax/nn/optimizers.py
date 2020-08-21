import numpy

import symjax
from symjax import tensor
from ..base import gradients
from symjax.nn.schedules import ExponentialMovingAverage


def conjugate_gradients(Ax, b):
    """
    Conjugate gradient algorithm
    (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
    """

    def ones_step():
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new

    x = T.zeros_like(b)
    r = (
        b.copy()
    )  # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
    p = r.copy()
    r_dot_old = np.dot(r, r)
    for _ in range(cg_iters):
        z = Ax(p)
        alpha = r_dot_old / (np.dot(p, z) + EPS)
        x += alpha * p
        r -= alpha * z
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / r_dot_old) * p
        r_dot_old = r_dot_new
    return x


class Optimizer:
    def __init__(self, *args, name=None, **kwargs):

        if name is None:
            name = self.__NAME__
        with symjax.Scope(name):
            self.create_updates(*args, **kwargs)

    def reset(self):
        if hasattr(self, "variables"):
            for var in self.variables:
                var.reset()

    @property
    def updates(self):
        if hasattr(self, "_updates"):
            return self._updates
        else:
            self._updates = {}
            return self._updates

    def _get_grads(self, grads_or_loss, params):
        # get grads if given is loss
        if (
            isinstance(grads_or_loss, tuple)
            or isinstance(grads_or_loss, list)
            or isinstance(grads_or_loss, tensor.MultiOutputOp)
        ):
            return grads_or_loss
        elif isinstance(grads_or_loss, tensor.Tensor):
            return gradients(grads_or_loss, params)
        else:
            return grads_or_loss

    def add_updates(self, update):
        if not hasattr(self, "_update"):
            self._updates = {}
        self._updates.update(update)
        symjax.current_graph().add_updates(update)


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

    __NAME__ = "SGDOptimizer"

    def create_updates(self, grads_or_loss, learning_rate, params=None):

        if params is None:
            params = symjax.get_variables(trainable=True)

        grads = self._get_grads(grads_or_loss, params)

        updates = dict()
        for param, grad in zip(params, grads):
            updates[param] = param - learning_rate * grad

        self.add_updates(updates)


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

    __NAME__ = "NesterovMomentumOptimizer"

    def create_updates(self, grads_or_loss, learning_rate, momentum, params=None):

        if params is None:
            params = symjax.get_variables(trainable=True)

        grads = self._get_grads(grads_or_loss, params)

        updates = dict()
        variables = []
        for param, grad in zip(params, grads):
            velocity = tensor.Variable(
                numpy.zeros(param.shape, dtype=param.dtype), trainable=False
            )
            variables.append(velocity)
            update = param - learning_rate * grad
            x = momentum * velocity + update - param
            updates[velocity] = x
            updates[param] = momentum * x + update

        self.add_updates(updates)


class Adam(Optimizer):
    """Adaptive Gradient Based Optimization with renormalization

    If amsgrad = False:
      Initialization:
      $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
      $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
      $$t := 0 \text{(Initialize timestep)}$$
      The update rule for `variable` with gradient `g` uses an optimization
      described at the end of section 2 of the paper:
      $$t := t + 1$$
      $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
      $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
      $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
      $$variable := variable - lr_t * m_t / (\sqrt{v_t} + \epsilon)$$
    If amsgrad = True:
      Initialization:
      $$m_0 := 0 \text{(Initialize initial 1st moment vector)}$$
      $$v_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
      $$v_hat_0 := 0 \text{(Initialize initial 2nd moment vector)}$$
      $$t := 0 \text{(Initialize timestep)}$$
      The update rule for `variable` with gradient `g` uses an optimization
      described at the end of section 2 of the paper:
      $$t := t + 1$$
      $$lr_t := \text{learning\_rate} * \sqrt{1 - beta_2^t} / (1 - beta_1^t)$$
      $$m_t := beta_1 * m_{t-1} + (1 - beta_1) * g$$
      $$v_t := beta_2 * v_{t-1} + (1 - beta_2) * g * g$$
      $$v_hat_t := max(v_hat_{t-1}, v_t)$$
      $$variable := variable - lr_t * m_t / (\sqrt{v_hat_t} + \epsilon)$$
    The default value of 1e-7 for epsilon might not be a good default in
    general. For example, when training an Inception network on ImageNet a
    current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
    formulation just before Section 2.1 of the Kingma and Ba paper rather than
    the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
    hat" in the paper.

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

    beta_1: constant or Tensor
        the value of the exponential moving average of the average of the
        gradients through time (updates)

    beta_2: constant or Tensor
        the value of the exponential moving average of the variance of the
        gradients through time

    Attributes
    ----------

    updates: list of updates

    variables: list of variables

    """

    __NAME__ = "AdamOptimizer"

    def create_updates(
        self,
        grads_or_loss,
        learning_rate,
        amsgrad=False,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        params=None,
    ):
        if params is None:
            params = symjax.get_variables(trainable=True)

        grads = self._get_grads(grads_or_loss, params)

        local_step = tensor.Variable(1, dtype="int32", trainable=False)
        updates = {local_step: local_step + 1}

        beta_1_t = tensor.power(beta_1, local_step)
        beta_2_t = tensor.power(beta_2, local_step)
        lr = learning_rate * (tensor.sqrt(1 - beta_2_t) / (1 - beta_1_t))

        for param, grad in zip(params, grads):
            m = ExponentialMovingAverage(grad, beta_1, debias=False)[0]
            v = ExponentialMovingAverage(grad ** 2, beta_2, debias=False)[0]
            if amsgrad:
                v_hat = tensor.Variable(
                    tensor.zeros_like(param), name="v_hat", trainable=False
                )
                updates[v_hat] = tensor.maximum(v_hat, v)
                update = m / (tensor.sqrt(updates[v_hat]) + epsilon)
            else:
                update = m / (tensor.sqrt(v) + epsilon)
            update = tensor.where(local_step == 1, grad, update)
            updates[param] = param - lr * update

        self.add_updates(updates)
