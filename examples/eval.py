import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import theanoxla.nn as nn

w = T.Placeholder((3000,1024), 'float32', name='w')
var = T.Variable(np.random.randn(1024, 1024), trainable=False, name='W')
step = T.Placeholder((),'float32', name='var')
u = T.sum(T.matmul(w, var)+step)

print(theanoxla.eval(u, {w: jax.numpy.ones((3000, 1024)),
                         step: jax.numpy.ones(1)}))

def fn(a, b):
    return theanoxla.eval(u, {w: a, step: b})

fnjit = jax.jit(fn)

import time
t = time.time()
for i in range(1000):
    theanoxla.eval(u, {w: jax.numpy.ones((3000, 1024)),
                         step: jax.numpy.ones(1)})
print(time.time()-t)


t = time.time()
for i in range(1000):
    fnjit(jax.numpy.ones((3000, 1024)), jax.numpy.ones(1))
print(time.time()-t)


t = time.time()
for i in range(1000):
    fnjit(np.ones((3000, 1024)), np.ones(1))
print(time.time()-t)

t = time.time()
for i in range(1000):
    fnjit(jax.numpy.array(np.ones((3000, 1024))), jax.numpy.array(np.ones(1)))
print(time.time()-t)










