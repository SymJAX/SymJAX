#!/usr/bin/env python
# -*- coding: utf-8 -*-

from symjax import tensor as T
from ..base import current_graph, Scope


def ExponentialMovingAverage(value, alpha):

    with Scope("ExponentialMovingAverage"):

        first_step = T.Variable(True, trainable=False, name="first_step", dtype="bool")

        var = T.Variable(
            T.zeros(value.shape), trainable=False, dtype="float32", name="EMA"
        )

        new_value = T.where(first_step, value, var * alpha + (1 - alpha) * value)

        current_graph().add({var: new_value, first_step: False})

    return new_value, var


def PiecewiseConstant(init, steps_and_values):

    with Scope("PiecewiseConstant"):

        all_steps = T.stack([0] + list(steps_and_values.keys()))
        all_values = T.stack([init] + list(steps_and_values.values()))

        step = T.Variable(T.zeros(1), trainable=False, name="step", dtype="float32",)

        value = all_values[(step < all_steps).argmin() - 1]

        current_graph().add({step: step + 1})

    return value
