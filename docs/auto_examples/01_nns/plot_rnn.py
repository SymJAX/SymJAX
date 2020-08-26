"""
RNN/GRU example
===========

example of vanilla RNN for time series regression
"""
import symjax.tensor as T
from symjax import nn
import symjax
import numpy as np
import matplotlib.pyplot as plt

symjax.current_graph().reset()
# create the network
BATCH_SIZE = 32
TIME = 32
WIDTH = 32
C = 1

np.random.seed(0)

timeseries = T.Placeholder((BATCH_SIZE, TIME, C), "float32", name="time-series")
target = T.Placeholder((BATCH_SIZE, TIME), "float32", name="target")

rnn = nn.layers.RNN(timeseries, np.zeros((BATCH_SIZE, WIDTH)), WIDTH)
rnn = nn.layers.RNN(rnn, np.zeros((BATCH_SIZE, WIDTH)), WIDTH)
rnn = nn.layers.Dense(rnn, 1, flatten=False)

gru = nn.layers.GRU(timeseries, np.zeros((BATCH_SIZE, WIDTH)), WIDTH)
gru = nn.layers.GRU(gru, np.zeros((BATCH_SIZE, WIDTH)), WIDTH)
gru = nn.layers.Dense(gru, 1, flatten=False)

loss = ((target - rnn[:, :, 0]) ** 2).mean()
lossg = ((target - gru[:, :, 0]) ** 2).mean()

lr = nn.schedules.PiecewiseConstant(0.01, {1000: 0.005, 1800: 0.001})

nn.optimizers.Adam(loss + lossg, lr)


train = symjax.function(
    timeseries,
    target,
    outputs=[loss, lossg],
    updates=symjax.get_updates(),
)

predict = symjax.function(timeseries, outputs=[rnn[:, :, 0], gru[:, :, 0]])


x = [
    np.random.randn(TIME) * 0.1 + np.cos(shift + np.linspace(-5, 10, TIME))
    for shift in np.random.randn(BATCH_SIZE * 200) * 0.3
]
w = np.random.randn(TIME) * 0.01
y = [(w + np.roll(xi, 2) * 0.4) ** 3 for xi in x]
y = np.stack(y)
x = np.stack(x)[:, :, None]
x /= np.linalg.norm(x, 2, 1, keepdims=True)
x -= x.min()
y /= np.linalg.norm(y, 2, 1, keepdims=True)


loss = []
for i in range(10):
    for xb, yb in symjax.data.utils.batchify(x, y, batch_size=BATCH_SIZE):
        loss.append(train(xb, yb))

loss = np.stack(loss)

plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.plot(loss[:, 0], c="g", label="Elman")
plt.plot(loss[:, 1], c="r", label="GRU")
plt.title("Training loss")
plt.xlabel("Iterations")
plt.ylabel("MSE")
plt.legend()

pred = predict(x[:BATCH_SIZE])

for i in range(4):
    plt.subplot(4, 2, 2 + 2 * i)

    plt.plot(x[i, :, 0], "-x", c="k", label="input")
    plt.plot(y[i], "-x", c="b", label="target")
    plt.plot(pred[0][i], "-x", c="g", label="Elman")
    plt.plot(pred[1][i], "-x", c="r", label="GRU")
    plt.title("Predictions")
    plt.legend()

plt.show()
