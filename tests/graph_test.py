#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

import symjax
import numpy as np


def test_reset():
    w = symjax.tensor.Variable(1.0, name="w", dtype="float32")
    x = symjax.tensor.Variable(2.0, name="x", dtype="float32")
    f = symjax.function(outputs=[w, x], updates={w: w + 1, x: x + 1})
    for i in range(10):
        print(i)
        assert np.array_equal(np.array(f()), np.array([1, 2]) + i)

    # reset only the w variable
    symjax.reset_variables("*w")
    assert np.array_equal(np.array(f()), np.array([1, 2 + i + 1]))
    # reset all variables
    symjax.reset_variables("*")
    assert np.array_equal(np.array(f()), np.array([1, 2]))


def test_accessing_variables():
    w1 = symjax.tensor.Variable(1.0, trainable=True)
    w2 = symjax.tensor.Variable(1.0, trainable=True)
    w3 = symjax.tensor.Variable(1.0, trainable=False)

    v = symjax.get_variables("*", trainable=True)
    assert w1 in v and w2 in v and w3 not in v

    v = symjax.get_variables("*", trainable=False)
    assert w1 not in v and w2 not in v and w3 in v

    v = symjax.get_variables("*test")
    assert len(v) == 0


def test_updating_variables():
    w1 = symjax.tensor.Variable(1.0, dtype="float32")
    input = symjax.tensor.Placeholder((), "float32")
    update = w1 + input + 1
    f = symjax.function(input, updates={w1: update})

    assert w1.value == 1.0
    f(10)
    assert w1.value == 12.0


def test_update():
    w = symjax.tensor.zeros(10)
    for i in range(10):
        w = symjax.tensor.index_update(w, i, i)
    f = symjax.function(outputs=w)
    assert np.array_equal(f(), np.arange(10))
    w2 = symjax.tensor.zeros(10)
    for i in range(10):
        w2 = symjax.tensor.index_update(w2, [i], i)
    f = symjax.function(outputs=w2)
    assert np.array_equal(f(), np.arange(10))

    w3 = symjax.tensor.zeros(10)
    for i in range(10):
        w3 = symjax.tensor.index_update(w3, symjax.tensor.index[i], i)
    f = symjax.function(outputs=w3)
    assert np.array_equal(f(), np.arange(10))

    w4 = symjax.tensor.Variable(symjax.tensor.zeros(10))
    i = symjax.tensor.Variable(0, dtype="int32")
    update = symjax.tensor.index_update(w4, i, i)
    f = symjax.function(updates={w4: update, i: i + 1})
    for i in range(10):
        f()
    assert np.array_equal(w4.value, np.arange(10))


if __name__ == "__main__":
    test_update()
    test_updating_variables()
    test_accessing_variables()
    test_reset()
