#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

import numpy as np

import symjax as sj
import symjax.tensor as T


def test_clone_0():
    sj.current_graph().reset()
    w = T.Variable(1.0, dtype="float32")
    with sj.Scope("placing"):
        u = T.Placeholder((), "float32", name="u")
    value = 2 * w * u
    c = value.clone({w: u})
    f = sj.function(u, outputs=value)
    g = sj.function(u, outputs=c)

    assert np.array_equal([f(1), g(1), f(2), g(2)], [2, 2, 4, 8])


def test_clone_base():
    sj.current_graph().reset()
    w = T.Variable(1.0, dtype="float32")
    w2 = T.Variable(1.0, dtype="float32")
    u = T.Placeholder((), "float32", name="u")
    uu = T.Placeholder((), "float32", name="uu")

    aa = T.Placeholder((), "float32")
    bb = T.Placeholder((), "float32")

    l = 2 * w * u * w2
    g = sj.gradients(l, w)
    guu = T.clone(l, {u: uu})
    guuu = T.clone(l, {w: uu})

    f = sj.function(u, outputs=g, updates={w2: w2 + 1})
    fuu = sj.function(uu, outputs=guu, updates={w2: w2 + 1})
    fuuu = sj.function(u, uu, outputs=guuu, updates={w2: w2 + 1})

    #    print(f(2))
    assert np.array_equal(f(2), 4.0)
    assert np.array_equal(fuu(1), 4)
    assert np.array_equal(fuuu(0, 0), 0)


if __name__ == "__main__":
    test_clone_base()
    test_clone_0()
