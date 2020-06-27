import sys

sys.path.insert(0, "../")

import symjax
import symjax.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

###### DERIVATIVE OF GAUSSIAN EXAMPLE


t = T.Placeholder((1000,), "float32")
print(t)
f = T.meshgrid(t, t)
f = T.exp(-(t ** 2))
u = f.sum()
g = symjax.gradients(u, [t])
g2 = symjax.gradients(g[0].sum(), [t])
g3 = symjax.gradients(g2[0].sum(), [t])

dog = symjax.function(t, outputs=[g[0], g2[0], g3[0]])

plt.plot(np.array(dog(np.linspace(-10, 10, 1000))).T)


###### GRADIENT DESCENT
z = T.Variable(3.0)
loss = z ** 2
g_z = symjax.gradients(loss, [z])
print(loss, z)
train = symjax.function(outputs=[loss, z], updates={z: z - 0.1 * g_z[0]})

losses = list()
values = list()
for i in range(5):
    a, b = train()
    losses.append(a)
    values.append(b)

plt.figure()
plt.subplot(121)
plt.plot(losses)
plt.subplot(122)
plt.plot(values, np.zeros_like(values), "kx")


###### NOISY GRADIENT DESCENT
z = T.Variable(3.0)
loss = z ** 2 + T.random.randn(()) * 10
g_z = symjax.gradients(loss, [z])
print(loss, g_z)
train = symjax.function(outputs=[loss, z], updates={z: z - 0.1 * g_z[0]})

losses = list()
values = list()
for i in range(10):
    a, b = train()
    losses.append(a)
    values.append(b)

plt.figure()
plt.subplot(121)
plt.plot(losses)
plt.subplot(122)
plt.plot(values, np.zeros_like(values), "kx")


####### jacobians

x, y = T.ones(()), T.ones(())
print(x, y)
ZZ = T.stack([x, y])
f = T.stack([3 * ZZ[0] + 2 * ZZ[1]], axis=0)
j = symjax.jacobians(f, [ZZ])[0]
g_j = symjax.function(outputs=j)


R = T.random.randn()
f = ZZ * 10 * R
j = symjax.jacobians(f, [ZZ])[0]
g_j = symjax.function(outputs=[j])
for i in range(5):
    print(g_j())


# plt.show()
