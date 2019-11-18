import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import theanoxla.nn as nn



w = T.Placeholder((3,), np.float32, name='w')

# MAP example 1
output = T.map(lambda a, b: T.pow(a, b), w, T.cast(T.arange(3), 'float32'))
print(output.get({w: jax.numpy.arange(3).astype('float32')}))
fn = theanoxla.function(w, outputs=[output])
print(fn(jax.numpy.arange(3).astype('float32')))

# MAP example 2
output = T.map(lambda a: T.pow(a, 2.), T.cast(T.arange(3), 'float32'))
print(output.get())
fn = theanoxla.function(outputs=[output])
print(fn())


# SCAN example 1
output = T.scan(lambda a, b: a + b, T.zeros(1), T.reshape(w, (3, 1)))
print(output.get({w: jax.numpy.arange(3).astype('float32')}))
fn = theanoxla.function(w, outputs=[output])
print(fn(jax.numpy.arange(3).astype('float32')))

# SCAN example 2
output = T.scan(lambda a, b, c: a + b * c, T.zeros(1),
                T.cast(T.arange(3), 'float32'), T.cast(T.arange(3), 'float32'))
print(output.get())
fn = theanoxla.function(outputs=[output])
print(fn())


# COND example 1
value = T.Placeholder((), np.float32)
output = T.cond(value < 0, (value, w), lambda a, b: a * b, (value, w),
                lambda a, b: a*b)
print(output.get({value: -1., w: jax.numpy.arange(3).astype('float32')}))
print(output.get({value: 1., w: jax.numpy.arange(3).astype('float32')}))
fn = theanoxla.function(value, w, outputs=[output])
print(fn(-1., jax.numpy.arange(3).astype('float32')))
print(fn(1., jax.numpy.arange(3).astype('float32')))

# COND example 2
value = T.Placeholder((), np.float32)
output = T.cond(value < 0, value, lambda a: a * 10, value,
                lambda a: a * 20)
print(output.get({value: -1.}))
print(output.get({value: 1.}))
fn = theanoxla.function(value, outputs=[output])
print(fn(-1.))
print(fn(1.))








