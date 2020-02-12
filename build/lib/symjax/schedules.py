#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import function
from . import tensor


class Schedule:

    def reset(self):
        for var in self.variables:
            var.reset()

    def update(self):
        if '_update' in self.__dict__:
            self._update()
        else:
            self._update = function(updates=self.updates)
            self._update()


class PiecewiseConstant(Schedule):

    def __init__(self, init, values):
        self.init = init
        self.values = values
        self.step = tensor.Variable(
            0, trainable=False, name='piecewise_constant_step')
        self.value = tensor.PiecewiseConstant(self.init, self.values,
                                              self.step)[0]
        self.updates = {self.step: self.step + 1}
        self.variables = [self.step]

    def __call__(self):
        return self.value


class Linear(Schedule):

    def __init__(self, init, slope):
        self.init = init
        self.slope = slope
        self.value = tensor.Variable(
            init, trainable=False, name='linear_variable')
        self.updates = {self.value: self.value + slope}
        self.variables = [self.value]

    def __call__(self):
        return self.value


class Exponential(Schedule):

    def __init__(self, init, slope):
        self.init = init
        self.slope = slope
        self.value = tensor.Variable(
            init, trainable=False, name='exponential_variable')
        self.updates = {self.value: self.value * slope}
        self.variables = [self.value]

    def __call__(self):
        return self.value
