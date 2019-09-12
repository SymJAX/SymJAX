import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T

SHAPE = (4, 4)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
w = T.Placeholder(SHAPE, 'float32', name='w')
y = T.cos(theanoxla.nn.activations.leaky_relu(z,0.3) + w)
cost = T.pool(y, (2, 2))
cost = T.sum(cost)

np.random.seed(10)
print(cost.get({w: np.random.randn(*SHAPE), z: z.value}))
np.random.seed(10)
print(cost.get({w: np.random.randn(*SHAPE)}))


fn = jax.jit(lambda x, y: cost.get({w: x, z:y}))
np.random.seed(10)
print(fn(np.random.randn(*SHAPE), z.value))
print(fn(np.random.randn(*SHAPE), z.value*1110))

fn1 = theanoxla.function(w, z, outputs=[cost])
np.random.seed(10)
print(fn1(np.random.randn(*SHAPE), z.value))
print(fn1(np.random.randn(*SHAPE), z.value*1110))
