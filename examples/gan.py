import sys
sys.path.insert(0, "../")

import numpy as np
import theanoxla.tensor as T
from theanoxla import layers, losses, optimizers, function, gradients
from theanoxla.utils import batchify, vq_to_boundary
from sklearn import datasets
import matplotlib.pyplot as plt


def generator(Z, out_dim):
    layer = [layers.Dense(Z, 16)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 16))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], out_dim))
    return layer

def discriminator(X):
    layer = [layers.Dense(X, 32)]
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 32))
    layer.append(layers.Activation(layer[-1], T.leaky_relu))
    layer.append(layers.Dense(layer[-1], 2))
    return layer


BS = 100
lr = 0.001
DATA, _ = datasets.make_moons(1000)

X = T.Placeholder([BS, 2], 'float32')
Z = T.Placeholder([BS, 2], 'float32')


G_sample = generator(Z, 2)
logits = discriminator(T.concatenate([G_sample[-1], X]))
labels = T.concatenate([T.zeros(BS, dtype='int32'), T.ones(BS, dtype='int32')])

disc_loss = losses.sparse_crossentropy_logits(labels, logits[-1]).mean()
gen_loss = losses.sparse_crossentropy_logits(1 - labels[:BS],
                                             logits[-1][:BS]).mean()
masks = T.concatenate([G_sample[1] > 0, G_sample[3] > 0], 1)

A = T.stack([gradients(G_sample[-1][:,0].sum(), [Z])[0],
             gradients(G_sample[-1][:,1].sum(), [Z])[0]], 1)
det = T.abs(T.det(A))

d_variables = sum([l.variables() for l in logits], [])
g_variables = sum([l.variables() for l in G_sample], [])

updates_d = optimizers.Adam(disc_loss, d_variables, lr)
updates_g = optimizers.Adam(gen_loss, g_variables, lr)
updates = {**updates_d, **updates_g}

f = function(Z, X, outputs = [disc_loss, gen_loss],
             updates = updates)
g = function(Z, outputs=[G_sample[-1]])

h = function(Z, outputs=[masks, det])

for epoch in range(3000):
    for x in batchify(DATA, batch_size=BS, option='random_see_all'):
        z = np.random.rand(BS, 2) * 2 -1
        f(z, x)

#
G = list()
for i in range(10):
        z = np.random.rand(BS, 2) * 2 -1
        G.append(g(z)[0])
G = np.concatenate(G)

#
NN = 300
xx, yy = np.meshgrid(np.linspace(-2, 2, NN), np.linspace(-2, 2, NN))
XX = np.stack([xx.flatten(), yy.flatten()], 1)
O = list()
O2 = list()
for x in batchify(XX, batch_size=BS, option='continuous'):
    a, b = h(x)
    O.append(a)
    O2.append(b)
O = np.concatenate(O)
O2 = np.log(np.concatenate(O2))
print(O2)
partition = vq_to_boundary(O, NN, NN)
partition_location = XX[partition.reshape((-1,)) > 0]

#
F = list()
for x in batchify(partition_location, batch_size=BS, option='continuous'):
    F.append(g(x)[0])
F = np.concatenate(F)

p2 = np.zeros((NN*NN,))
for i in range(len(F)):
    distances = np.abs(XX - F[i]).max(1)
    istar = distances.argmin()
    if distances[istar] <= 6 / NN:
        p2[istar] = 1

#
H1 = list()
H2 = list()
for i in range(7):
    time = np.linspace(-1, 1, 200)
    H1.append(np.stack([time, time*(np.random.rand()*4-2)], 1))
    H2.append([])
    for x in batchify(H1[-1], batch_size=BS, option='continuous'):
        H2[-1].append(g(x)[0])
    H2[-1] = np.concatenate(H2[-1])


p2 = p2.reshape((NN, NN))

plt.subplot(241)
plt.imshow(partition, aspect='auto', origin='lower', extent=(-2, 2, -2, 2))
plt.title('z space partition')

plt.subplot(242)
plt.imshow(O2.reshape((NN, NN)), aspect='auto', origin='lower', extent=(-2, 2, -2, 2))
plt.title('z space A determinant')

plt.subplot(243)
plt.imshow(p2, aspect='auto', origin='lower', extent=(-2, 2, -2, 2))
plt.title('x space corresponding partition')

plt.subplot(244)
plt.plot(G[:, 0], G[:, 1], 'x')
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.title('generated points')

plt.subplot(246)
for i in range(6):
    plt.plot(H1[i][:,0], H1[i][:,1])

plt.subplot(248)
for i in range(6):
    plt.plot(H2[i][:,0], H2[i][:,1])


plt.show(block=True)

