#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RNTK kernel
===========

tiem series regression and classification
"""
import numpy as np
import symjax
import symjax.tensor as T


def RNTK_first_time_step(x, param):
    # this is for computing the first GP and RNTK for t = 1. Both for relu and erf
    sw = param["sigmaw"]
    su = param["sigmau"]
    sb = param["sigmab"]
    sh = param["sigmah"]
    X = x * x[:, None]
    print(X)
    n = X.shape[0]
    GP_new = sh ** 2 * sw ** 2 * T.eye(n, n) + (su ** 2 / m) * X + sb ** 2
    RNTK_new = GP_new
    return RNTK_new, GP_new


def RNTK_relu(x, RNTK_old, GP_old, param, output):
    sw = param["sigmaw"]
    su = param["sigmau"]
    sb = param["sigmab"]
    sv = param["sigmav"]

    a = T.diag(GP_old)  # GP_old is in R^{n*n} having the output gp kernel
    # of all pairs of data in the data set
    B = a * a[:, None]
    C = T.sqrt(B)  # in R^{n*n}
    D = GP_old / C  # this is lamblda in ReLU analyrucal formula
    # clipping E between -1 and 1 for numerical stability.
    E = T.clip(D, -1, 1)
    F = (1 / (2 * np.pi)) * (E * (np.pi - T.arccos(E)) + T.sqrt(1 - E ** 2)) * C
    G = (np.pi - T.arccos(E)) / (2 * np.pi)
    if output:
        GP_new = sv ** 2 * F
        RNTK_new = sv ** 2.0 * RNTK_old * G + GP_new
    else:
        X = x * x[:, None]
        GP_new = sw ** 2 * F + (su ** 2 / m) * X + sb ** 2
        RNTK_new = sw ** 2.0 * RNTK_old * G + GP_new
    return RNTK_new, GP_new


L = 10
N = 3
DATA = T.Placeholder((N, L), "float32")
# parameters
param = {}
param["sigmaw"] = 1.33
param["sigmau"] = 1.45
param["sigmab"] = 1.2
param["sigmah"] = 0.4
param["sigmav"] = 2.34
m = 1

# first time step
RNTK, GP = RNTK_first_time_step(DATA[:, 0], param)

for t in range(1, L):
    RNTK, GP = RNTK_relu(DATA[:, t], RNTK, GP, param, False)

RNTK, GP = RNTK_relu(0, RNTK, GP, param, True)


f = symjax.function(DATA, outputs=[RNTK, GP])


# three data of length T
a = np.random.randn(L)
b = np.random.randn(L)
c = np.random.randn(L)
example = np.stack([a, b, c])  # it is of shape (3, T)
print(f(example))
