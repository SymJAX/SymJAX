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
from sklearn.metrics import roc_auc_score, accuracy_score




# dataset
wavs, labels, infos = theanoxla.datasets.load_freefield1010(subsample=2, n_samples=1000)
wavs /= wavs.max(1, keepdims=True)
wavs_train, wavs_test, labels_train, labels_test = theanoxla.utils.train_test_split(wavs, labels, 0.33)

# variables
BS = 16
signal = T.Placeholder((BS, len(wavs[0])), 'float32')
label = T.Placeholder((BS,), 'int32')
deterministic = T.Placeholder((1,), 'bool')

# first layer

#mixing = T.signal.spectrogram(T.expand_dims(signal, 1), 512, hop=64)
mixing = T.signal.wvd2(T.expand_dims(signal, 1), 512, L=8, hop=32)
layer = []
layer.append(layers.Conv2D(mixing, 1, (32, 32), strides=(2,2)))
layer.append(layers.Conv2D(T.log(T.abs(layer[-1])+ 0.0001), 16, (3, 3)))
#layer.append(layers.Conv2D(mixing, 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (3, 3)))

layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (3, 3)))

layer.append(layers.Conv2D(layer[-1], 16, (3, 1)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (3, 1)))

layer.append(layers.Conv2D(layer[-1], 16, (3, 1)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
#layer.append(layers.Pool2D(layer[-1], (3, 1)))

layer.append(layers.Dense(layer[-1], 256))
layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

layer.append(layers.Dense(layer[-1], 32))
layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

layer.append(layers.Dense(T.relu(layer[-1]), 2))

loss = theanoxla.losses.sparse_crossentropy_logits(label, layer[-1])
accuracy = theanoxla.losses.accuracy(label, layer[-1])
var = sum([lay.variables for lay in layer], [])

grads = theanoxla.gradients(loss, var)
lr = theanoxla.optimizers.PiecewiseConstantSchedule(0.001, {15000: 0.0003, 30000: 0.0001})
updates = theanoxla.optimizers.Adam(grads, var, lr)
for lay in layer:
    updates.update(lay.updates)


f = theanoxla.function(signal, label, deterministic, outputs = [loss, accuracy],
                       updates=updates)
g = theanoxla.function(signal, label, deterministic, outputs = [T.softmax(layer[-1])[:,1], accuracy])
h = theanoxla.function(signal, outputs=[mixing])
#getcov = theanoxla.function(outputs=[COV])
getgrads= theanoxla.function(signal, label, outputs = grads)


for epoch in range(100):
    L = list()
    for x, y in theanoxla.utils.batchify(wavs_train, labels_train, batch_size=BS,
                                         option='random_see_all'):
        L.append(f(x, y, 0)[1])
#        VV = h(x)[0]
#        plt.imshow(VV[0,0], aspect='auto')
#        plt.show(block=True)
    print(np.mean(L))
    L = list()
    C = list()
    for x, y in theanoxla.utils.batchify(wavs_test, labels_test, batch_size=BS,
                                         option='continuous'):
        a, c = g(x, y, 1)
        L.append(a)
        C.append(c)
    L = np.concatenate(L)
    print('FINAL', np.mean(C), roc_auc_score(labels_test[:len(L)], L))



