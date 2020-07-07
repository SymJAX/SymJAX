"""
RNN example
===========

example of vanilla RNN for time series regression
"""
import symjax.tensor as T
from symjax import nn
import symjax
import numpy as np
import matplotlib.pyplot as plt

# create the network
BATCH_SIZE = 1
TIME = 128
C = 1


timeseries = T.Placeholder((BATCH_SIZE, TIME, C), "float32", name="time-series")
target = T.Placeholder((BATCH_SIZE, TIME), "float32", name="target")

layer1 = nn.layers.RNN(timeseries, np.zeros((BATCH_SIZE, 3)), 3)
layer2 = nn.layers.RNN(layer1, np.zeros((BATCH_SIZE, 1)), 1)


loss = ((layer2[:, :, 0] - target) ** 2).mean()

nn.optimizers.Adam(loss, 0.05)


train = symjax.function(timeseries, target, outputs=loss, updates=symjax.get_updates(),)


x = np.random.randn(TIME) * 0.1 + np.cos(np.linspace(0, 10, TIME))
y = np.convolve(x, np.random.randn(5), mode="same").reshape((1, -1))
x = x.reshape((1, -1, 1))

loss = []
for i in range(100):
    loss.append(train(x, y))


plt.plot(loss)
plt.title("Training loss")
plt.xlabel("Iterations")
plt.ylabel("MSE")
