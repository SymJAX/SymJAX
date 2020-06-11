#!/usr/bin/env python
# -*- coding: utf-8 -*-

import symjax as sj
import symjax.tensor as T
import jax.numpy as jnp

__author__      = "Randall Balestriero"
import tensorflow_probability as tfp


tfp = tfp.experimental.substrates.jax

nn = tfp.distributions.Normal(1., 5.)

print(nn.cdf(1))

mean = T.Placeholder((1,), 'float32', name='mean')
normal_dist = T.wrap_class(tfp.distributions.Normal)
a = normal_dist(mean, 5.)

x = T.Variable(T.ones(1)-5)
output = a.cdf(x)

get_f = sj.function(mean, outputs=output)

for i in range(-5, 5):
    print('mean:',1,' x:', T.get(x), ' cdf:', get_f(i))


