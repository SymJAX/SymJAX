#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"


import symjax
import numpy as np


def test_pc():
    a, cpt = symjax.nn.schedules.PiecewiseConstant(0, {4: 1, 8: 2})
    f = symjax.function(outputs=a, updates={cpt: cpt + 1})
    for i in range(10):
        value = f()
        if i < 4:
            assert np.array_equal(value, 0)
        elif i < 8:
            assert np.array_equal(value, 1)
        else:
            assert np.array_equal(value, 2)


if __name__ == "__main__":
    test_pc()
