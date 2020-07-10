"""
Basic gradient descent (and reset)
==================================

demonstration on how to compute a gradient and apply a basic gradient update
rule to minimize some loss function

"""

import symjax
import symjax.tensor as T
import matplotlib.pyplot as plt

# GRADIENT DESCENT
z = T.Variable(3.0, dtype="float32")
loss = (z - 1) ** 2
g_z = symjax.gradients(loss, [z])[0]
symjax.current_graph().add_updates({z: z - 0.1 * g_z})

train = symjax.function(outputs=[loss, z], updates=symjax.get_updates())

losses = list()
values = list()
for i in range(200):
    if (i + 1) % 50 == 0:
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
