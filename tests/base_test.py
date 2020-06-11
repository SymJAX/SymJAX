#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Randall Balestriero"


import symjax
import numpy as np
import symjax.tensor as tt

import jax
import jax.numpy as jnp
import jax.scipy as jsp

def test_add():
    a = symjax.tensor.ones(2)
    assert symjax.tensor.get(a.max()) == 1

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
            tau = self.sigma ** -2.
            return (-tau * (value - self.mu)**2 + tt.log(tau / np.pi / 2.)) / 2.
    
        def random(self, sample_shape):
            return np.random.randn(sample_shape) * self.sigma + self.mu
    
    x = Normal('x', 0, 10.)
    s = Normal('s', 0., 5.)
    y = Normal('y', x, tt.exp(s))

    assert tt.get(y) == 0.

    #################
    model_logpt = x.logpt + s.logpt + y.logpt

    f = symjax.function(x, s, y, outputs=model_logpt)

    normal_loglike = jsp.stats.norm.logpdf

    f_ = lambda x, s, y: (normal_loglike(x, 0., 10.) +
                      normal_loglike(s, 0., 5.) +
                      normal_loglike(y, x, jnp.exp(s)))


    for i in range(10):
        x_val = np.random.randn() * 10.
        s_val = np.random.randn() * 5.
        y_val = np.random.randn() * .1 + x_val

        np.testing.assert_allclose(f(x_val, s_val, y_val), f_(x_val, s_val, y_val), rtol=1e-06)

    model_dlogpt = symjax.gradients(model_logpt, [x, s, y])

    f_with_grad = symjax.function(x, s, y, outputs=[model_logpt, model_dlogpt])

    f_with_grad(x_val, s_val, y_val)

    grad_fn = jax.grad(f_, argnums=[0, 1, 2])
    f_(x_val, s_val, y_val), grad_fn(x_val, s_val, y_val)
