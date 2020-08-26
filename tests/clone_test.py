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


def test_clone_1():
    sj.current_graph().reset()
    w = T.Variable(1.0, dtype="float32")
    ww = T.Variable(1.0, dtype="float32")
    x = T.Placeholder((), "float32")
    z = x * 2
    loss = z * w * ww + 1 + 2 + 3
    grad = sj.gradients(loss, [w])[0]
    loss2 = loss.clone({z: x})
    grad2 = sj.gradients(loss2, [w])[0]

    f = sj.function(x, outputs=grad)
    g = sj.function(x, outputs=grad2)

    assert np.array_equal(f(3), g(6))
    assert np.array_equal(f(3), 6.0)

    ww.update(2.0)

    assert np.array_equal(f(3), g(6))
    assert np.array_equal(f(3), 12.0)


def test_clone_2():
    sj.current_graph().reset()
    w = T.Variable(1.0, dtype="float32")
    ww = T.Variable(1.0, dtype="float32")
    x = T.Placeholder((100,), "float32")
    y = T.Placeholder((100,), "float32")
    loss = T.sum(x * w + ww)
    other = loss.clone({x: y})
    grad = sj.gradients(other, [w, ww])

    f = sj.function(y, outputs=grad)

    assert np.array_equal(f(np.ones(100) * 2)[1], 100.0)
    assert np.array_equal(f(np.ones(100) * 2)[0], 200.0)


def test_clone_3():
    sj.current_graph().reset()
    w = T.Variable(T.ones(10), dtype="float32")
    z = T.Variable(T.ones(3), dtype="float32")

    x = T.Placeholder((10,), "float32")
    y = T.Placeholder((3,), "float32")

    v1 = x * w
    total = v1.sum()

    v2 = v1.clone({x: y, w: z})
    print(v2)
    total2 = v2.sum()

    f = sj.function(x, y, outputs=[total, total2])

    assert np.array_equal(f(np.ones(10), np.ones(3)), [10, 3])


def test_clone_4():
    sj.current_graph().reset()
    a = T.ones((10, 1))
    b = T.ones((10, 1))
    c = T.concatenate([a, b], 1)
    arg, kwargs = sj.current_graph().get_args_kwargs(c, evaluate=False)

    w = T.Variable(T.ones((10, 1)))
    d = c.clone({a: 2 * w})

    f = sj.function(outputs=[c, d])

    assert np.array_equal(f()[0], np.ones((10, 2)))
    assert np.array_equal(
        f()[1], np.concatenate([2 * np.ones((10, 1)), np.ones((10, 1))], 1)
    )
    w.update(np.ones((10, 1)) * 4)
    assert np.array_equal(f()[0], np.ones((10, 2)))
    assert np.array_equal(
        f()[1], np.concatenate([8 * np.ones((10, 1)), np.ones((10, 1))], 1)
    )


def test_clone_5():
    sj.current_graph().reset()
    a = T.ones((10,))
    b = T.random.randn((10,))
    c = a.dot(b)
    d = c.clone({a: 2 * a})

    f = sj.function(outputs=[c, d])

    outs = f()
    assert np.allclose(outs[0] * 2, outs[1])
    outs2 = f()
    assert np.allclose(outs2[0] * 2, outs2[1])
    assert not np.allclose(outs[0], outs2[0])


def test_clone_6():
    sj.current_graph().reset()
    a = T.ones(20).astype("float32")
    final, each = T.scan(lambda c, i: (c + i, c + i), sequences=[a], init=float(0))
    assert each.shape.get() == (20,)
    each2 = each.clone({a: a * 4})
    assert each2.shape.get() == (20,)
    f = sj.function(outputs=[each, each2])
    assert np.array_equal(f()[0], np.arange(20) + 1)
    assert np.array_equal(f()[1], (np.arange(20) + 1) * 4)


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
    test_clone_1()
    test_clone_2()
    test_clone_3()
    test_clone_4()
    test_clone_5()
    test_clone_6()
