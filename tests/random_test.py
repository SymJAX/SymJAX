#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"


import symjax
import symjax.tensor as T
import numpy as np


def test_base():
    a = T.random.randn(())
    f = symjax.function(outputs=a)
    print([f() for i in range(100)])


def test_seed():
    a = T.random.randn((), seed=10)
    b = T.random.randn(())
    c = T.random.randn((), seed=10)
    f = symjax.function(outputs=[a, b, c])
    result1 = f()
    result2 = f()
    print(result1)
    print(result2)
    assert result1[0] == result1[2]
    assert result1[0] != result1[1]

    assert result2[0] == result2[2]
    assert result2[0] != result1[0]

    a = T.random.randn((), seed=10)
    b = T.random.randn(())
    c = T.random.randn((), seed=10)
    f = symjax.function(outputs=[a, b, c])
    result12 = f()
    result22 = f()
    assert result12[0] == result12[2]
    assert result12[0] != result12[1]
    assert result22[0] == result22[2]
    assert result22[0] != result12[0]

    assert np.isclose(result1[0], result12[0])
    assert np.isclose(result1[2], result12[2])
    assert not np.isclose(result1[1], result12[1])

    assert np.isclose(result2[0], result22[0])
    assert np.isclose(result2[2], result22[2])
    assert not np.isclose(result2[1], result22[1])

    symjax.current_graph().reset()

    a = T.random.randn((), seed=10)
    b = T.random.randn(())
    c = T.random.randn((), seed=10)
    f = symjax.function(outputs=[a, b, c])
    result12 = f()
    result22 = f()
    assert result12[0] == result12[2]
    assert result12[0] != result12[1]
    assert result22[0] == result22[2]
    assert result22[0] != result12[0]

    assert np.isclose(result1[0], result12[0])
    assert np.isclose(result1[2], result12[2])
    assert not np.isclose(result1[1], result12[1])

    assert np.isclose(result2[0], result22[0])
    assert np.isclose(result2[2], result22[2])
    assert not np.isclose(result2[1], result22[1])


if __name__ == "__main__":
    test_base()
    test_seed()
