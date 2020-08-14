#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"


import symjax
import numpy as np
import symjax.tensor as tt
import jax
import jax.numpy as jnp
import jax.scipy as jsp


def test_add():
    a = symjax.tensor.ones(2)
    assert symjax.current_graph().get(a.max()) == 1


def test_ema():
    symjax.current_graph().reset()
    a = symjax.tensor.Placeholder((), "float32")
    ema, var = symjax.nn.schedules.ExponentialMovingAverage(a, 0.9, debias=False)
    # t = symjax.get_variables("*num_steps*", trainable=False)
    f = symjax.function(a, outputs=[ema, var], updates=symjax.get_updates())
    current = 0

    for i in range(10):
        out = f(1)
        assert np.allclose(out[1], current)
        current = 0.9 * current + 0.1 * 1
        assert np.allclose(out[0], current)


def test_stop():
    a = symjax.tensor.ones(())
    b = a + a ** 2
    g = symjax.gradients(b, [a])[0]
    f = symjax.function(outputs=g)
    assert f() == 3
    b = a + symjax.tensor.stop_gradient(a ** 2)
    g = symjax.gradients(b, [a])[0]
    f = symjax.function(outputs=g)
    assert f() == 1


def test_g():
    a = symjax.tensor.ones(())
    b = symjax.tensor.Variable(1.0)
    l = a * b
    g = symjax.gradients(l, [a])[0]
    f = symjax.function(outputs=g, updates={b: b + 1.0})
    assert f() == 1
    assert f() == 2
    assert f() == 3


def test_pymc():
    class RandomVariable(symjax.tensor.Variable):
        def __init__(self, name, shape, observed):
            if observed is None:
                super().__init__(np.zeros(shape), name=name)
            else:
                super().__init__(observed, name=name, trainable=False)

        def logp(self, value):
            raise NotImplementedError()

        def random(self, sample_shape):
            raise NotImplementedError()

        @property
        def logpt(self):
            return self.logp(self)

    class Normal(RandomVariable):
        def __init__(self, name, mu, sigma, shape=None, observed=None):
            self.mu = mu
            self.sigma = sigma
            super().__init__(name, shape, observed)

        def logp(self, value):
            tau = self.sigma ** -2.0
            return (-tau * (value - self.mu) ** 2 + tt.log(tau / np.pi / 2.0)) / 2.0

        def random(self, sample_shape):
            return np.random.randn(sample_shape) * self.sigma + self.mu

    x = Normal("x", 0, 10.0)
    s = Normal("s", 0.0, 5.0)
    y = Normal("y", x, tt.exp(s))

    assert symjax.current_graph().get(y) == 0.0

    #################
    model_logpt = x.logpt + s.logpt + y.logpt

    f = symjax.function(x, s, y, outputs=model_logpt)

    normal_loglike = jsp.stats.norm.logpdf

    def f_(x, s, y):
        return (
            normal_loglike(x, 0.0, 10.0)
            + normal_loglike(s, 0.0, 5.0)
            + normal_loglike(y, x, jnp.exp(s))
        )

    for i in range(10):
        x_val = np.random.randn() * 10.0
        s_val = np.random.randn() * 5.0
        y_val = np.random.randn() * 0.1 + x_val

        np.testing.assert_allclose(
            f(x_val, s_val, y_val), f_(x_val, s_val, y_val), rtol=1e-06
        )

    model_dlogpt = symjax.gradients(model_logpt, [x, s, y])

    f_with_grad = symjax.function(x, s, y, outputs=[model_logpt, model_dlogpt])

    f_with_grad(x_val, s_val, y_val)

    grad_fn = jax.grad(f_, argnums=[0, 1, 2])
    f_(x_val, s_val, y_val), grad_fn(x_val, s_val, y_val)


def test_grad():
    w = tt.Placeholder((), "float32")
    v = tt.Variable(1.0, dtype="float32")
    x = w * v + 2
    symjax.nn.optimizers.Adam(x, 0.001)
    g = symjax.gradients(x, [v])
    f = symjax.function(w, outputs=g[0])
    assert f(1) == 1
    assert f(10) == 10


if __name__ == "__main__":
    test_grad()
    test_add()
    test_pymc()
    test_stop()
    test_g()
    test_ema()
