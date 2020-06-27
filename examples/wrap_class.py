#!/usr/bin/env python
# -*- coding: utf-8 -*-

import symjax as sj
import symjax.tensor as T
import jax.numpy as jnp

__author__ = "Randall Balestriero"


class product:
    def __init__(self, W, V=1):
        self.W = jnp.square(V * W * (W > 0).astype("float32"))
        self.ndim = self.compute_ndim()

    def feed(self, x):
        return jnp.dot(self.W, x)

    def compute_ndim(self):
        return self.W.shape[0] * self.W.shape[1]


wrapped = T.wrap_class(product, method_exceptions=["compute_ndim"])


a = wrapped(T.zeros((10, 10)), V=T.ones((10, 10)))
x = T.random.randn((10, 100))

print(a.W)
# (Tensor: name=function[0], shape=(10, 10), dtype=float32)

print(a.feed(x))
# Op(name=feed, shape=(10, 100), dtype=float32, scope=/)

f = sj.function(outputs=a.feed(x))

f()
