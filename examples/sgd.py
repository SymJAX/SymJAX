import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import numpy as np


SHAPE = (4, 4)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
z2 = T.Variable(np.ones(SHAPE).astype('float32'), name='z2')

cost = T.sum(T.pool(T.cos(z + z2), (2, 2)))
grads = theanoxla.gradients(cost, [z])

sgd = theanoxla.optimizers.SGD([z], grads, 0.001)
adam = theanoxla.optimizers.Adam([z], grads, 0.001)
getgrad = theanoxla.function(outputs=[grads[0]], updates={z2:z2+1})

trainsgd = theanoxla.function(outputs=[cost], updates=sgd)
trainadam = theanoxla.function(outputs=[cost], updates=adam)


for i in range(10):
    print(getgrad())
exit()

cost = list()
for i in range(1000):
    cost.append(trainsgd()[0])

z.reset()

for i in range(1000):
    cost.append(trainadam()[0])



import matplotlib.pyplot as plt
plt.plot(cost)
plt.title('cost')
plt.show()
