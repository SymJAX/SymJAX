import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import theanoxla.nn as nn

SHAPE = (4, 4)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
w = T.Placeholder(SHAPE, 'float32', name='w')
noise = T.random.bernoulli(SHAPE, p=0.8, dtype='float32')
noise2 = T.random.uniform(SHAPE, dtype='float32')
y = T.cos(theanoxla.nn.activations.leaky_relu(z,0.3) + w + noise + T.cast(noise2 > 0.5, 'float32'))
cost = T.sum(T.square(y-1))
grads = theanoxla.gradients(cost, [z])
updates = theanoxla.nn.optimizers.SGD([z], grads, 0.01)

train = theanoxla.function(w, outputs=[cost], updates=updates)

import matplotlib.pyplot as plt
cost = list()
for i in range(1000):
    cost.append(train(np.ones((4, 4)))[0])
plt.plot(cost)
plt.show()
