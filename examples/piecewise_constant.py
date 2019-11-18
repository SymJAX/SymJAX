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



BS = 1


value, step = T.PiecewiseConstant(0, {10:1, 20:2})
f = theanoxla.function(outputs=[value], updates={step:step+1})

for epoch in range(100):
    print(f())



