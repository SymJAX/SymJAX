#!/usr/bin/env python
# -*- coding: utf-8 -*-

import symjax as sj
import jax.numpy as jnp

__author__ = "Randall Balestriero"

# suppose we want to compute the mean-squared error between two vectors
x = sj.tensor.random.normal((10,))
y = sj.tensor.zeros((10,))

# one way is to do so by combining SymJAX functions as
mse = ((x - y) ** 2).sum()
# notice that the basic operators are overloaded and implicitly call SymJAX ops

# another solution is to create a new SymJAX Op from a jax computation as
# follows


def mse_jax(x, y):
    return jnp.sum((x - y) ** 2)


# wrap the jax computation into a SymJAX Op that can then be used as any
# SymJAX function
mse_op = sj.tensor.jax_wrap(mse_jax)
also_mse = mse_op(x, y)
print(also_mse)
# Tensor(Op=mse_jax, shape=(), dtype=float32)


# ensure that both are equivalent
f = sj.function(outputs=[mse, also_mse])
print(f())
# [array(6.0395503, dtype=float32), array(6.0395503, dtype=float32)]
