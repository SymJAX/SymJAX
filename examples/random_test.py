import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T

# test the random normal/uniform
SHAPE = (2, 2)
randn = T.random.randn(SHAPE)
rand = T.random.rand(SHAPE)

get_vars = theanoxla.function(outputs=[randn, rand])

for i in range(10):
    print(get_vars())

# test shuffle
matrix = T.linspace(0, 1, 16).reshape((4, 4))
smatrix = T.random.shuffle(matrix, 0)

get_shuffle = theanoxla.function(outputs=smatrix)

for i in range(10):
    print(get_shuffle())


asdfsadf

# test the random uniform
SHAPE = (2, 2)
z = T.Variable(np.random.randn(*SHAPE).astype('float32'), name='z')
get_z = theanoxla.function(outputs=z)

for i in range(10):
    print(get_z())






#print(f_shuffle()[0])
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
