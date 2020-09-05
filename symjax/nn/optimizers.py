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

    def _get_variables(self, loss):

        params = symjax.get_variables(trainable=True)

        print(params)
        params = [p for p in params if symjax.current_graph().is_connected(p, loss)]
        print(params)
        return params

    def add_updates(self, update):
        if not hasattr(self, "_update"):
            self._updates = {}
        self._updates.update(update)
        symjax.current_graph().add_updates(update)


class SGD(Optimizer):
    """Stochastic gradient descent optimization.

    Notice that SGD is also the acronym employed in ``tf.keras.optimizers.SGD``
    and in ``torch.optim.sgd`` but might be misleading. In fact, those
    and this implementation implement GD, the SGD term only applies if one
    performs GD optimization only using 1 (random) sample to compute the gradients.
    If multiple samples are used it is commonly referred as mini-batch GD and
    when the entire dataset is used then the optimizer is refered as GD. See
    an illustrative discussion `here <https://towardsdatascience.com/difference-between-batch-gradient-descent-and-stochastic-gradient-descent-1187f1291aa1>`_.

    The produced update for parameter θ and a given learning rate α is:

    .. math::
        θ = θ - α  ∇_{θ} L

    Parameters
    ----------

    grads_or_loss: scalar tensor or list of gradients
        either the loss (scalar of Tensor type) to be differentied
        or the list of gradients already computed and possibly altered
        manually (such as clipping)

    learning_rate: constant or Tensor
        the learning rate use to update the parameters

    params: list (optional)
        if grads_or_loss is al list then it should be ordered w.r.t. the
        given parameters

    Attributes
    ----------

    updates: list of updates

    variables: list of variables

    """

    __NAME__ = "SGDOptimizer"

    def create_updates(self, grads_or_loss, learning_rate, params=None):

        if isinstance(grads_or_loss, list):
            assert params

        if params is None:
            params = self._get_variables(grads_or_loss)
        elif type(params) != list:
            raise RuntimeError("given params should be a list")

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

    learning_rate: constant or Tensor
        the learning rate use to update the parameters

    momentum: constant or Tensor
        the amount of momentum to be applied

    params: list (optional)
        if grads_or_loss is al list then it should be ordered w.r.t. the
        given parameters


    Attributes
    ----------

    updates: list of updates

    variables: list of variables

    """

    __NAME__ = "NesterovMomentumOptimizer"

    def create_updates(self, grads_or_loss, learning_rate, momentum, params=None):

        if isinstance(grads_or_loss, list):
            assert params
        if params is None:
            params = self._get_variables(grads_or_loss)
        elif type(params) != list:
            raise RuntimeError("given params should be a list")

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
    """Adaptive Gradient Based Optimization with renormalization.

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section 2 of the paper with learning rate
    α.

    If ``amsgrad`` is ``False``:

    **initialization**:

        - :math:`m_0 = 0` (Initialize initial 1st moment vector)
        - :math:`v_0 = 0` (Initialize initial 2nd moment vector)
        - :math:`t = 0` (Initialize timestep)

    **update**:

        - :math:`t = t + 1`
        - :math:`α_t = α × \sqrt{1 - β_2^t}/(1 - β_1^t)`
        - :math:`m_t = β_1 × m_{t-1} + (1 - β_1) × g`
        - :math:`v_t = β_2 × v_{t-1} + (1 - β_2) × g \odot g`
        - :math:`variable = variable - α_t × m_t / (\sqrt{v_t} + ε)`

    If ``amsgrad`` is ``True``:

    **initialization**:

        - :math:`m_0 = 0` (Initialize initial 1st moment vector)
        - :math:`v_0 = 0` (Initialize initial 2nd moment vector)
        - :math:`v'_0 = 0` (Initialize initial 2nd moment vector)
        - :math:`t = 0` (Initialize timestep)

    **update**:

        - :math:`t = t + 1`
        - :math:`α_t = α × \sqrt{1 - β_2^t}/(1 - β_1^t)`
        - :math:`m_t = β_1 × m_{t-1} + (1 - β_1) × g`
        - :math:`v_t = β_2 × v_{t-1} + (1 - β_2) × g \odot g`
        - :math:`v'_t := \max(v'_{t-1}, v_t)`
        - :math:`variable = variable - α_t × m_t / (\sqrt{v'_t} + ε)`

    The default value of :math:`\epsilon=1e-7` might not be a good default in
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

    learning_rate (α): constant or Tensor
        the learning rate use to update the parameters

    amsgrad: bool
        whether to use the amsgrad updates or not

    β_1: constant or Tensor
        the value of the exponential moving average of the average of the
        gradients through time (updates)

    β_2: constant or Tensor
        the value of the exponential moving average of the variance of the
        gradients through time

    ε : constant or Tensor
        the value added to the second order moment

    params: list (optional)
        if grads_or_loss is al list then it should be ordered w.r.t. the
        given parameters, if not given then the optimizer will find
        all variables that are traininable and involved with the
        given loss

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

        if isinstance(grads_or_loss, list):
            assert params
        if params is None:
            params = self._get_variables(grads_or_loss)
        elif type(params) != list:
            raise RuntimeError("given params should be a list")

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
