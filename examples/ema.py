import jax
import numpy as np
import sys
sys.path.insert(0, "../")

import theanoxla
import theanoxla.tensor as T
import theanoxla.nn as nn

w = T.Placeholder((3,), 'float32', name='w')
t = T.Placeholder((), 'float32', name='t')
update = T.cond(t==0, T.List([w]), T.sum(w), T.List([t]), t*4)
fn = theanoxla.function(t, w, outputs=[update])

print(fn(3.,jax.numpy.ones(3)))
print(fn(2.,jax.numpy.ones(3)))
print(fn(3.,jax.numpy.ones(3)))

print('OUTOUT', update.get({t:jax.numpy.array(3.), w:jax.numpy.ones(3)}))
print('OUTOUT', update.get({t:jax.numpy.array(2.), w:jax.numpy.ones(3)}))
print('OUTOUT', update.get({t:jax.numpy.array(3.), w:jax.numpy.ones(3)}))
#exit()
#################
alpha = 0.5
var, updates, step = nn.ExponentialMovingAverage(w, alpha)
train = theanoxla.function(w, outputs=[updates[var]], updates=updates)

import matplotlib.pyplot as plt
data = np.stack([np.ones(20), np.random.randn(20), np.zeros(20)], 1)
cost = list()
true_ema = [data[0]]
aa= 0.5
for j, i in enumerate(data):
    cost.append(train(i)[0])
    true_ema.append(aa*true_ema[-1]+(1-aa)*i)
    print(cost[-1])
cost = np.asarray(cost)
true = np.asarray(true_ema)[1:]
print(np.mean(np.isclose(cost, true)))

plt.subplot(131)
plt.plot(data[:, 0])
plt.plot(cost[:, 0])
plt.plot(true[:, 0])

plt.subplot(132)
plt.plot(data[:, 1])
plt.plot(cost[:, 1])
plt.plot(true[:, 1])

plt.subplot(133)
plt.plot(data[:, 2])
plt.plot(cost[:, 2])
plt.plot(true[:, 2])

plt.show()
