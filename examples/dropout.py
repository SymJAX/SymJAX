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
# https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client
from sklearn.metrics import roc_auc_score, accuracy_score


BS = 1
signal = T.Placeholder((BS, 4), "float32")
deterministic = T.Placeholder((1,), "bool")
random = T.random.bernoulli((2, 2), p=0.5)
output = layers.Dropout(signal, 0.5, deterministic)
g = theanoxla.function(outputs=[random])
f = theanoxla.function(signal, deterministic, outputs=[output])

for epoch in range(100):
    print(g())

for epoch in range(100):
    print(f(np.ones((BS, 4)), 0)[0])

for epoch in range(100):
    print(f(np.ones((BS, 4)), 1)[0])
