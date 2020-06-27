import time
import pickle
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
# https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client
from sklearn.metrics import roc_auc_score, accuracy_score

import argparse

parse = argparse.ArgumentParser()
parse.add_argument("-L", type=int)
args = parse.parse_args()

# dataset
wavs, labels, infos = theanoxla.datasets.load_freefield1010(subsample=2, n_samples=7000)
wavs /= wavs.max(1, keepdims=True)
wavs_train, wavs_test, labels_train, labels_test = theanoxla.utils.train_test_split(
    wavs, labels, 0.33
)

# variables
L = args.L
BS = 6


signal = T.Placeholder((BS, len(wavs[0])), "float32")

if L > 0:
    WVD = T.signal.wvd(T.expand_dims(signal, 1), 1024, L=L, hop=32)
else:
    WVD = T.signal.mfsc(
        T.expand_dims(signal, 1), 1024, 192, 80, 2, 44100 / 4, 44100 / 4
    )

tf_func = theanoxla.function(signal, outputs=[WVD], backend="cpu")

tf = T.Placeholder(WVD.shape, "float32")
label = T.Placeholder((BS,), "int32")
deterministic = T.Placeholder((1,), "bool")

# first layer
NN = 32
if L > 0:
    x, y, = T.meshgrid(T.linspace(-5, 5, NN), T.linspace(-5, 5, NN))
    grid = T.stack([x.flatten(), y.flatten()], 1)
    cov = T.Variable(np.eye(2), name="cov")
    gaussian = T.exp(-(grid.dot(cov.T().dot(cov)) * grid).sum(1)).reshape(
        (1, 1, NN, NN)
    )
    layer = [layers.Conv2D(tf, 1, (NN, NN), strides=(6, 6), W=gaussian, b=None)]
    layer[-1].add_variable(cov)
    layer.append(layers.Activation(layer[-1], lambda x: T.log(T.abs(x) + 0.01)))
else:
    layer = [layers.Activation(tf + 0.01, T.log)]

layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (3, 3)))

layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (3, 3)))

layer.append(layers.Conv2D(layer[-1], 16, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Pool2D(layer[-1], (1, 2)))

layer.append(layers.Conv2D(layer[-1], 32, (3, 3)))
layer.append(layers.BatchNormalization(layer[-1], [0, 2, 3], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
# layer.append(layers.Pool2D(layer[-1], (3, 1)))

layer.append(layers.Dense(layer[-1], 256))
layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Dropout(layer[-1], 0.5, deterministic))

layer.append(layers.Dense(layer[-1], 32))
layer.append(layers.BatchNormalization(layer[-1], [0], deterministic))
layer.append(layers.Activation(layer[-1], T.leaky_relu))
layer.append(layers.Dropout(layer[-1], 0.2, deterministic))

layer.append(layers.Dense(T.relu(layer[-1]), 2))

loss = theanoxla.losses.sparse_crossentropy_logits(label, layer[-1])
accuracy = theanoxla.losses.accuracy(label, layer[-1])
var = sum([lay.variables for lay in layer], [])

lr = theanoxla.optimizers.PiecewiseConstantSchedule(
    0.001, {15000: 0.0003, 30000: 0.0001}
)
updates = theanoxla.optimizers.Adam(loss, var, lr)
for lay in layer:
    updates.update(lay.updates)

f = theanoxla.function(
    tf, label, deterministic, outputs=[loss, accuracy], updates=updates
)
g = theanoxla.function(
    tf, label, deterministic, outputs=[T.softmax(layer[-1])[:, 1], accuracy]
)


# transform the data
tf_train = list()
tf_test = list()

for x in theanoxla.utils.batchify(wavs_train, batch_size=BS, option="continuous"):
    tf_train.append(tf_func(x)[0])
tf_train = np.concatenate(tf_train, 0)

for x in theanoxla.utils.batchify(wavs_test, batch_size=BS, option="continuous"):
    tf_test.append(tf_test(x)[0])
tf_test = np.concatenate(tf_test, 0)


DATA = []
for epoch in range(100):
    l = list()
    for x, y in theanoxla.utils.batchify(
        tf_train, labels_train, batch_size=BS, option="random_see_all"
    ):
        l.append(f(x, y, 0))
        print(l[-1])
    DATA.append(l)
    l = list()
    C = list()
    for x, y in theanoxla.utils.batchify(
        tf_test, labels_test, batch_size=BS, option="continuous"
    ):
        a, c = g(x, y, 1)
        l.append(a)
        C.append(c)
    l = np.concatenate(l)
    DATA.append((np.mean(C), roc_auc_score(labels_test[: len(l)], l)))
    print(
        "FINAL",
        np.mean(C),
        roc_auc_score(labels_test[: len(l)], (l > 0.5).astype("int32")),
    )
    ff = open("saveit_{}.pkl".format(L), "wb")
    pickle.dump(DATA, ff)
    ff.close()
