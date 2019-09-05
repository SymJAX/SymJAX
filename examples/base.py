import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T

key = jax.random.PRNGKey(1)
z = T.Variable(np.random.normal(1, 1))


w = T.Placeholder((1, 1), 'float32')
y = T.cos(theanoxla.nn.activations.relu(z) + w)
cost = T.sum(y)

grad = theanoxla.gradients(cost, [w, z])

train = theanoxla.function(w, outputs=[cost],
                 updates={z:z-grad[1]*0.01})

for i in range(10):
    print(train(np.ones((1, 1))))


