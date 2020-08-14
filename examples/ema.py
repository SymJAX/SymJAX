import matplotlib.pyplot as plt
import symjax.tensor as T
import symjax
import numpy as np


w = T.Placeholder((3,), "float32", name="w")
alpha = 0.5
new_value, var = symjax.nn.schedules.ExponentialMovingAverage(w, alpha)

train = symjax.function(w, outputs=new_value, updates=symjax.get_updates())


data = np.stack([np.ones(200), np.random.randn(200), np.zeros(200)], 1)
cost = list()
true_ema = [data[0]]
aa = 0.5
for j, i in enumerate(data):
    cost.append(train(i))
    true_ema.append(aa * true_ema[-1] + (1 - aa) * i)
cost = np.asarray(cost)
true = np.asarray(true_ema)[1:]
print("% close values:", 100 * np.mean(np.isclose(cost, true)))


plt.subplot(311)
plt.plot(data[:, 0])
plt.plot(cost[:, 0])
plt.plot(true[:, 0])

plt.subplot(312)
plt.plot(data[:, 1], label="true")
plt.plot(cost[:, 1], label="symjax")
plt.plot(true[:, 1], label="np")
plt.legend()


plt.subplot(313)
plt.plot(data[:, 2], label="true")
plt.plot(cost[:, 2], label="symjax")
plt.plot(true[:, 2], label="np")

plt.legend()
plt.show()
