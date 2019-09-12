import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T

SHAPE = (4, 4)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
w = T.Placeholder(SHAPE, 'float32', name='w')
noise = T.random.uniform(SHAPE, dtype='float32')
y = T.cos(theanoxla.nn.activations.leaky_relu(z,0.3) + w + noise)
cost = T.pool(y, (2, 2))
cost = T.sum(cost)

print(cost.get({w: np.random.randn(*SHAPE)}))
noise.seed = 20
print(cost.get({w: np.random.randn(*SHAPE)}))
noise.seed = 40
print(cost.get({w: np.random.randn(*SHAPE)}))


#grad = theanoxla.gradients(cost, [w, z])
#train = theanoxla.function(w, outputs=[cost, noise, noise2, T.cast(noise2 > 0.5, 'float32')],
#                 updates={z:z-grad[1]*0.01})

#cost = list()
#for i in range(1000):
#    cost.append(train(np.ones((4, 4)))[0])

#import matplotlib.pyplot as plt
#plt.plot(cost)
#plt.show()
