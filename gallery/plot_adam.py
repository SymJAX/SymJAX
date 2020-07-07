"""
Basic Adam update (and reset)
=============================

demonstration on how to use Adam optimizer to minimize a loss

"""

import symjax
import symjax.tensor as T
from symjax.nn.optimizers import Adam
import matplotlib.pyplot as plt

# GRADIENT DESCENT
z = T.Variable(3.0, dtype="float32", trainable=True)
loss = T.power(z - 1, 2, name="loss")
print(loss)
Adam(loss, 0.1)

train = symjax.function(outputs=[loss, z], updates=symjax.get_updates())

losses = list()
values = list()
for i in range(200):
    if (i + 1) % 100 == 0:
        symjax.reset_variables("*")
    a, b = train()
    losses.append(a)
    values.append(b)

plt.figure()

plt.subplot(121)
plt.plot(losses, "-x")
plt.ylabel("loss")
plt.xlabel("number of gradient updates")

plt.subplot(122)
plt.plot(values, "-x")
plt.axhline(1, c="red")
plt.ylabel("value")
plt.xlabel("number of gradient updates")

plt.tight_layout()
plt.show()
