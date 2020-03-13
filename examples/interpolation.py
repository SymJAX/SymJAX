#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../")      
import symjax as sj
import symjax.tensor as T
import matplotlib.pyplot as plt
import numpy as np

__author__ = "Randall Balestriero"

J = 8
Q = 1
K = 9
M = K * 2 ** (J - 1)

scales = T.power(2,T.arange(J*Q) / Q)
time = T.arange(M) - M // 2
knots = (T.arange(K) - K // 2) * scales.reshape((-1, 1))
 
values = T.cos(np.pi * T.range(K)) * T.signal.hamming(K)
derivatives = T.cos(np.pi * T.range(K)) * T.diff(T.signal.hamming(K+1))

mask = T.ones((K,)) * (1-T.one_hot(0, K)-T.one_hot(K-1, K))
spline = sj.interpolation.hermite(time, knots, values * mask,
        derivatives * mask)

spline = T.signal.hilbert_transform(spline)

spline_r = spline/T.linalg.norm(spline, 2, 1, keepdims=True)

f = sj.function(outputs=spline_r)

for filt in f()[[-1]]:
    plt.plot(np.real(filt))
    plt.plot(np.imag(filt))


plt.show()
