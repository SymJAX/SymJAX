import time
import jax
import numpy as np
import sys
sys.path.insert(0, "../")
from scipy.io.wavfile import read

import theanoxla
import theanoxla.tensor as T

import matplotlib.pyplot as plt
from matplotlib import interactive
interactive(False)
#https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client

def wvd(signal, h, hop):
    p = T.extract_signal_patches(signal, h, hop)
    pr = p * T.flip(p, 3)
    qr = T.real(T.fft(T.cast(p,'complex64'), xla_client.FftType.FFT, (h,)))
    return T.transpose(qr[..., :h], [0, 1, 3, 2])


wavs, labels, infos = theanoxla.datasets.load_freefield1010(n_samples=300, subsample=2)
#wavs = np.random.randn(100, 2**14)
#labels = np.random.randint(0, 2, 100)
wavs /= wavs.max(1, keepdims=True)

BS = 16
signal = T.Placeholder((BS, len(wavs[0])), 'float32')
label = T.Placeholder((BS,), 'int32')

layer1 = wvd(T.expand_dims(signal, 1), 4096, hop=64)


X, Y = T.meshgrid(T.linspace(-5, 5, 32), T.linspace(-5, 5, 32))
Z = T.stack([X.flatten(), Y.flatten()], 1)
COV = T.Variable(np.random.rand(2,2), name='cov')
gaussian = T.exp(-(Z.dot(T.abs(COV))*Z).sum(1)).reshape((32, 32))

filter1 = T.expand_dims(T.expand_dims(gaussian, 0), 0) / gaussian.sum()
layer2 = T.log(T.abs(T.convNd(layer1, filter1, strides=(1, 30)))+0.0001)

layer25 = T.poolNd(layer2, (1,1,1,512))

FILTER1 = T.Variable(np.random.randn(16, 1, 3, 3))
BIAS1 = T.Variable(np.random.randn(1, 16, 1, 1))
layer3 = T.abs(T.convNd(layer2, FILTER1, strides=(1, 2)) + BIAS1)

FILTER2 = T.Variable(np.random.randn(32, 16, 3, 3))
BIAS2 = T.Variable(np.random.randn(1, 32, 1, 1))
layer4 = T.abs(T.convNd(layer3, FILTER2, strides=(1, 2)) + BIAS2)


FILTER3 = T.Variable(np.random.randn(32, 2))
BIAS3 = T.Variable(np.random.randn(2))
layer5 = T.dot(layer4.mean((2, 3)), FILTER3) + BIAS3

loss = theanoxla.losses.sparse_crossentropy_logits(label, layer5)
accuracy = theanoxla.losses.accuracy(label, layer5)

var = [COV, FILTER1, FILTER2, FILTER3, BIAS1, BIAS2, BIAS3]
grads = theanoxla.gradients(loss, var)

updates = theanoxla.optimizers.Adam(var, grads, 0.0001)

f = theanoxla.function(signal, label, outputs = [loss, accuracy],
                       updates=updates)
getcov = theanoxla.function(outputs=[COV])
getgrads= theanoxla.function(signal, label, outputs = grads)
for i in range(100):
    indices = np.random.permutation(len(wavs))[:BS]
    print(f(wavs[indices], labels[indices]))
    print(COV.get({}))



