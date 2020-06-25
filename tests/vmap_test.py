#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import symjax


def test_vectorize():
    x = symjax.tensor.Placeholder((0, 2), "float32")
    w = symjax.tensor.Variable(1.0, dtype="float32")
    p = x.sum(1)

    f = symjax.function(x, outputs=p, updates={w: x.sum()})

    assert np.array_equal(f(np.ones((1, 2))), [2.0])
    assert w.value == 2.0
    assert np.array_equal(f(np.ones((2, 2))), [2.0, 2.0])
    assert w.value == 4.0


def test_vectorize_sgd():
    x = symjax.tensor.Placeholder((0, 2), "float32")
    y = symjax.tensor.Placeholder((0,), "float32")

    w = symjax.tensor.Variable((1, 1), dtype="float32")
    loss = ((x.dot(w) - y) ** 2).mean()

    g = symjax.gradients(loss, [w])[0]

    other_g = symjax.gradients(x.dot(w).sum(), [w])[0]

    f = symjax.function(x, y, outputs=loss, updates={w: w - 0.1 * g})
    other_f = symjax.function(x, outputs=other_g)

    L = [10]
    for i in range(10):
        L.append(f(np.ones((i + 1, 2)), -1 * np.ones(i + 1)))
        assert L[-1] < L[-2]
        assert np.array_equal(other_f(np.ones((i + 1, 2))), [i + 1.0, i + 1.0])


if __name__ == "__main__":
    test_vectorize_sgd()
    test_vectorize()
