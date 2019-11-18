import time
import jax
import numpy as np
import sys
sys.path.insert(0, "../")
from scipy.io.wavfile import read

import theanoxla
import theanoxla.tensor as T
from theanoxla import layers

import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(False)
#https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client



images, labels = theanoxla.datasets.load_cifar10()

images_train, images_test = images
labels_train, labels_test = labels
images_train /= images_train.max((1, 2, 3), keepdims=True)
images_test /= images_test.max((1, 2, 3), keepdims=True)


BS = 32
inputs = T.Placeholder((BS,) + images_train.shape[1:], 'float32')
outputs = T.Placeholder((BS,), 'int32')
deterministic = T.Placeholder((1,), 'bool')

layer = [layers.Conv2D(inputs, 32, (3, 3))]
for i in range(3):
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Conv2D(T.relu(layer[-1]), 32, (3, 3)))
for i in range(3):
    layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
    layer.append(layers.Conv2D(T.relu(layer[-1]), 64, (3, 3)))

layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Dense(layer[-1], 10))
#layer.append(layers.Dense(T.relu(layer[-1]), 10))

#layer = [layers.Dense(inputs, 10)]
loss = theanoxla.losses.sparse_crossentropy_logits(outputs, layer[-1])
accuracy = theanoxla.losses.accuracy(outputs, layer[-1])

params = sum([[lay.W, lay.b] for lay in layer], [])

updates = theanoxla.optimizers.Adam(loss, params, 0.001)

for l in layer:
    updates.update(l.updates)


g = theanoxla.function(inputs, outputs, deterministic, outputs = [loss, accuracy])

f = theanoxla.function(inputs, outputs, deterministic, outputs = [loss, accuracy],
                       updates=updates)

print(updates)
for epoch in range(100):
    L = list()
    for x, y in theanoxla.utils.batchify(images_test, labels_test, batch_size=BS,
                                         option='continuous'):
        L.append(g(x, y, 1)[1])
    print(np.mean(L))
#        print(layer[0].W.get({}))
    L = list()
    for x, y in theanoxla.utils.batchify(images_train, labels_train, batch_size=BS,
                                         option='random_see_all'):
        L.append(f(x, y, 0)[1])
    print('FINAL', np.mean(L))
