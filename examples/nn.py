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
cost1 = nn.Dense(y, 1)
cost = T.sum(T.square(cost1-1))
var = cost1.variables(trainable=True)
params = [z]+var

print('asdfasdfasdf')
print(params)
print(cost.all_dependencies)
#exit()
grads = theanoxla.gradients(cost, params)
lr = T.Placeholder((), 'float32', name='lr')
updates = theanoxla.nn.optimizers.SGD(params, grads, lr)

train = theanoxla.function(w, lr, outputs=[cost, noise, noise2, T.cast(noise2 > 0.5, 'float32')],
                 updates=updates)

import matplotlib.pyplot as plt
cost = list()
for i in range(1000):
    cost.append(train(np.ones((4, 4)), 0.0001)[0])
plt.plot(cost)
plt.show()
