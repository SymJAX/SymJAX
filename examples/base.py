import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T

z = T.Variable(np.random.normal(1, 1), name='z')
w = T.Placeholder((1, 1), 'float32', name='w')
noise = T.random.bernoulli((1, 1), p=0.8, dtype='float32')
noise2 = T.random.uniform((1, 1), dtype='float32')
y = T.cos(theanoxla.nn.activations.leaky_relu(z,0.3) + w + noise + T.cast(noise2 > 0.5, 'float32'))
cost = T.sum(y)

grad = theanoxla.gradients(cost, [w, z])
train = theanoxla.function(w, outputs=[cost, noise, noise2, T.cast(noise2 > 0.5, 'float32')],
                 updates={z:z-grad[1]*0.01})

for i in range(10):
    print(train(np.ones((1, 1))))


