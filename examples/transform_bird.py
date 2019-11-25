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
#https://github.com/google/jax/blob/master/jax/lib/xla_bridge.py
from jax.lib import xla_client
from sklearn.metrics import roc_auc_score, accuracy_score

import argparse

parse = argparse.ArgumentParser()
parse.add_argument('-L', type=int)
args = parse.parse_args()

# dataset
wavs, labels, infos = theanoxla.datasets.load_freefield1010(subsample=2, n_samples=7000)

# variables
L = args.L
BS = 1


signal = T.Placeholder((BS, wavs.shape[1]), 'float32')

if L > 0:
    WVD = T.signal.wvd(T.expand_dims(signal, 1), 1024, L=L, hop=32)
else:
    WVD = T.signal.mfsc(T.expand_dims(signal, 1), 1024, 192, 80, 2, 44100/4, 44100/4)

tf_func = theanoxla.function(signal, outputs=[WVD], backend='cpu')

# transform the data
tf_train = list()
print('start')

for x in theanoxla.utils.batchify(wavs, batch_size=BS,
                                  option='continuous'):
    print('before')
    tf_train.append(tf_func(x)[0])
    print('sdf')

tf_train = np.concatenate(tf_train, 0)
print('save')
np.savez_compressed('new_bird_{}.npz'.format(L), tfs=tf_train, labels=labels)

