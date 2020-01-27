import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T


o = T.one_hot(1, 5)
print(o.get({}))

SHAPE = (2, 2)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')

# shuffle
z_1 = T.upsample(z, (2, 1), 'repeat')
z_2 = T.upsample(z, (1, 2), 'repeat')
z_3 = T.upsample(z, (2, 3), 'repeat')

f_shuffle = theanoxla.function(outputs=[z, z_1, z_2, z_3])
A = f_shuffle()
for a in A:
    print(a)

# shuffle
zz_1 = T.upsample(z, (2, 1))
zz_2 = T.upsample(z, (1, 2))
zz_3 = T.upsample(z, (2, 3))

ff_shuffle = theanoxla.function(outputs=[z, zz_1, zz_2, zz_3])
A = ff_shuffle()
for a in A:
    print(a)





asdasd












w = T.Placeholder(SHAPE, 'float32', name='w')
noise = T.random.uniform(SHAPE, dtype='float32')
y = T.cos(theanoxla.nn.activations.leaky_relu(z,0.3) + w + noise)
cost = T.pool(y, (2, 2))
cost = T.sum(cost)

grads = theanoxla.gradients(cost, [w, z], [1])

print(cost.get({w: np.random.randn(*SHAPE)}))
noise.seed = 20
print(cost.get({w: np.random.randn(*SHAPE)}))
noise.seed = 40
print(cost.get({w: np.random.randn(*SHAPE)}))

updates = {z:z-0.01*grads[0]}
fn1 = theanoxla.function(w, outputs=[cost])
fn2 = theanoxla.function(w, outputs=[cost], updates=updates)
print(fn1(np.random.randn(*SHAPE)))
print(fn1(np.random.randn(*SHAPE)))

cost = list()
for i in range(1000):
    cost.append(fn2(np.ones(SHAPE))[0])

import matplotlib.pyplot as plt
plt.plot(cost)
plt.show()
