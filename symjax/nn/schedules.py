#!/usr/bin/env python
# -*- coding: utf-8 -*-

from symjax import tensor as T
from ..base import current_graph, Scope
import numpy as np


def ExponentialMovingAverage(value, alpha, init=None):

    """exponential moving average of a given value

    This method allows to obtain an EMA of a given variable (or any Tensor)
    with internal state automatically upating its values as new samples are
    observed and the internal updates are applied as part of a fuction
    At each iteration the new value is given by

    .. math::

      v(0) = value(0) or init
      v(t) = v(t-1) * alpha + value(t) * (1 - alpha)

    Args
    ----

    value: Tensor-like
        the value to use for the EMA

    alpha: scalar
        the decay of the EMA

    init: Tensor-like (same shape as value) optional
        the initialization of the EMA, if not given uses the value
        allowing for unbiased estimate

    Returns
    -------

    ema: Tensor-like
        the current (latest) value of the EMA incorporating information
        of the latest observation of value

    fixed_ema: Tensor-like
        the value of the EMA of the previous pass. This is usefull if one wants
        to keep the estimate of the EMA fixed for new observations, then simply
        do not apply anymore updates (using a new function) and using this
        fixed variable during testing (while ema will keep use the latest 
        observed value)



    Example
    -------

    .. doctest ::
    >>> import symjax
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> symjax.current_graph().reset()
    >>> # suppose we want to do an EMA of a vector user-input
    >>> input = symjax.tensor.Placeholder((2,), 'float32')
    >>> ema, var = symjax.nn.schedules.ExponentialMovingAverage(input, 0.9)
    >>> # in the background, symjax automatically records the needed updates
    >>> print(symjax.get_updates())
    {Variable(name=EMA, shape=(2,), dtype=float32, trainable=False, scope=/ExponentialMovingAverage/): Op(name=where, fn=where, shape=(2,), dtype=float32, scope=/ExponentialMovingAverage/), Variable(name=first_step, shape=(), dtype=bool, trainable=False, scope=/ExponentialMovingAverage/): False}
    >>> # example of use:
    >>> f = symjax.function(input, outputs=ema, updates=symjax.get_updates())
    >>> for i in range(25):
    ...     print(f(np.ones(2) + np.random.randn(2) * 0.3))
    [1.5292157 1.1200472]
    [1.5056562 1.1752692]
    [1.5111173 1.1284239]
    [1.4885082 1.1110408]
    [1.4365609 1.1122546]
    [1.3972261 1.1446574]
    [1.3803346 1.1338419]
    [1.355617  1.1304679]
    [1.3648777 1.1112664]
    [1.3377819 1.0745169]
    [1.227414  1.0866737]
    [1.2306056 1.0557414]
    [1.2756376 1.0065362]
    [1.2494465 1.000267 ]
    [1.2704852 1.0443211]
    [1.2480851 1.0512339]
    [1.196643  0.9866866]
    [1.1665413 0.9927084]
    [1.186796 1.029509]
    [1.1564965 1.017489 ]
    [1.1093903  0.97313946]
    [1.0472631 1.0343488]
    [1.0272473 1.0177717]
    [0.9869387 1.0393193]
    [0.93982786 1.029005  ]



    """

    with Scope("ExponentialMovingAverage"):

        init = init if init is not None else value

        first_step = T.Variable(True, trainable=False, name="first_step", dtype="bool")

        var = T.Variable(
            T.zeros(value.shape), trainable=False, dtype="float32", name="EMA"
        )

        new_value = T.where(first_step, init, var * alpha + (1 - alpha) * value)

        current_graph().add_updates({var: new_value, first_step: False})

    return new_value, var


def PiecewiseConstant(init, steps_and_values):
    """piecewise constant variable updating automatically

    This method allows to obtain a variable with an internal counter
    that will be updated based on the function updates, whenver this
    counter reaches one of the step given in the function input
    then the actual value of the variable becomes the one given for the
    associated step

    Args
    ----

    init: float-like
        the initial value of the variable that will remain as is until a step
        and update is reached

    steps_and_values: dict
        the dictionnary mapping steps-> values, that is, when the number of
        steps reached one of the given one, the value of the variable becomes
        the given one associated to the reached step

    Returns
    -------

    variable: float-like

    Example
    -------

    .. doctest ::
    >>> import symjax
    >>> symjax.current_graph().reset()
    >>> var = symjax.nn.schedules.PiecewiseConstant(0.1, {4:1, 8:2})
    >>> # in the background, symjax automatically records that everytime
    >>> # a function is using this variable an udnerlying update should occur
    >>> print(symjax.get_updates())
    {Variable(name=step, shape=(), dtype=int32, trainable=False, scope=/PiecewiseConstant/): Op(name=add, fn=add, shape=(), dtype=int32, scope=/PiecewiseConstant/)}
    >>> # it is up to the user to use it or not, if not used, the internal counter
    >>> # is never updated and this the variable never changes.
    >>> # example of use:
    >>> f = symjax.function(outputs=var, updates=symjax.get_updates())
    >>> for i in range(10):
    ...     print(i, f())
    0 0.1
    1 0.1
    2 0.1
    3 0.1
    4 1.0
    5 1.0
    6 1.0
    7 1.0
    8 2.0
    9 2.0

    """

    with Scope("PiecewiseConstant"):

        all_steps = T.stack([0] + list(steps_and_values.keys()) + [np.inf])
        all_values = T.stack([init] + list(steps_and_values.values()) + [0])

        step = T.Variable(0, trainable=False, name="step", dtype="int32",)

        value = all_values[(step >= all_steps).argmin() - 1]

        current_graph().add_updates({step: step + 1})

    return value
