#!/usr/bin/env python
# -*- coding: utf-8 -*-

from symjax import tensor
import numpy
from ..base import function


class Schedule:
    def reset(self):
        for var in self.variables:
            var.reset()

    def update(self):
        if "_update" in self.__dict__:
            self._update()
        else:
            self._update = function(updates=self.updates)
            self._update()


def piecewise_constant(init, values, step=None):
    """
    init it the initial value
    values is a dictionnary mapping the knots and the new value
    step count the number of steps
    """
    if step is None:
        step = tensor.Variable(0, trainable=False, name="PiecewiseConstant_step")
    keys, values = list(values.keys()), list(values.values())
    keys.insert(0, 0)
    values.insert(0, init)
    keys, values = numpy.array(keys), numpy.array(values)
    assert numpy.prod(keys >= 0)
    arg = numpy.argsort(keys)
    keys, values = keys[arg], values[arg]
    index = (step < keys).argmax()
    v = tensor.Variable(values, trainable=False, name="PiecewiseConstant_values")
    return tensor.dynamic_slice_in_dim(v, index - 1, 1, 0), step


class PiecewiseConstant(Schedule):
    def __init__(self, init, values):
        self.init = init
        self.values = values
        self.step = tensor.Variable(
            0, trainable=False, name="piecewise_constant_step", dtype="float32"
        )
        self.value = piecewise_constant(self.init, self.values, self.step)[0]
        self.updates = {self.step: self.step + 1}
        self.variables = [self.step]

    def __call__(self):
        return self.value


class Linear(Schedule):
    def __init__(self, init, slope):
        self.init = init
        self.slope = slope
        self.value = tensor.Variable(init, trainable=False, name="linear_variable")
        self.updates = {self.value: self.value + slope}
        self.variables = [self.value]

    def __call__(self):
        return self.value


def ExponentialMovingAverage(value, alpha, step=None, init=None):
    if step is None:
        _step = tensor.Variable(0, trainable=False, name="step", dtype="float32")
    else:
        _step = step
    if init is None:
        var = tensor.Variable(
            numpy.zeros(value.shape), trainable=False, name="EMA", dtype="float32"
        )
    else:
        var = tensor.Variable(init, trainable=False, name="EMA", dtype="float32")

    new_value = tensor.where(
        tensor.equal(_step, 0), value, var * alpha + (1 - alpha) * value
    )
    if step is None:
        updates = {var: new_value, _step: _step + 1}
    else:
        updates = {var: new_value}
    return var, updates, _step


class Exponential(Schedule):
    def __init__(self, init, slope):
        self.init = init
        self.slope = slope
        self.value = tensor.Variable(
            init, trainable=False, name="exponential_variable", dtype="float32"
        )
        self.updates = {self.value: self.value * slope}
        self.variables = [self.value]

    def __call__(self):
        return self.value
