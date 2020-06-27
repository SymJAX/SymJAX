import jax
import numpy as np
import sys

sys.path.insert(0, "../")

import symjax
import symjax.tensor as T

w = T.Placeholder((3,), "float32", name="w")
alpha = 0.5
var, updates, step = T.ExponentialMovingAverage(w, alpha)

train = symjax.function(w, outputs=[updates[var]], updates=updates)


import matplotlib.pyplot as plt

data = np.stack([np.ones(20), np.random.randn(20), np.zeros(20)], 1)
cost = list()
true_ema = [data[0]]
aa = 0.5
for j, i in enumerate(data):
    cost.append(train(i)[0])
    true_ema.append(aa * true_ema[-1] + (1 - aa) * i)
cost = np.asarray(cost)
true = np.asarray(true_ema)[1:]
print("% close values:", 100 * np.mean(np.isclose(cost, true)))

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
