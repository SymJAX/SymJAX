import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import numpy as np


SHAPE = (4, 4)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
w = T.ones(SHAPE, name='w')

cost = T.sum(T.pool(T.cos(z + w), (2, 2)))

grads = theanoxla.gradients(cost, [w, z], [1])

train = theanoxla.function(outputs=[cost], updates={z:z-0.03*grads[0]})

cost = list()
for i in range(1000):
    cost.append(train()[0])

import matplotlib.pyplot as plt
plt.plot(cost)
plt.title('cost')
plt.show()
