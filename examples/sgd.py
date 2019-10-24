import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import numpy as np


SHAPE = (1, 1)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
z2 = T.Variable(np.ones(SHAPE).astype('float32'), name='z2')

cost = T.sum(T.poolNd(T.cos(z + z2), (1, 1)))
grads = theanoxla.gradients(cost, [z])
print('gradients', grads)
sgd = theanoxla.optimizers.SGD([z], grads, 0.001)
adam = theanoxla.optimizers.Adam([z], grads, 0.001)

#getgrad = theanoxla.function(outputs=[grads[0]], updates={z2:z2+1})
trainsgd = theanoxla.function(outputs=[cost], updates=sgd)
trainadam = theanoxla.function(outputs=[cost], updates=adam)


#for i in range(10):
#    print(getgrad())
#exit()

cost = list()
for i in range(10):
    cost.append(trainsgd()[0])
    print(z.get({}))

z.reset()

for i in range(10):
    cost.append(trainadam()[0])



import matplotlib.pyplot as plt
plt.plot(cost)
plt.title('cost')
plt.show(block=True)
