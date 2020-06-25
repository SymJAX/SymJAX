#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import time


class Identity:
    def __init__(self, eps=0.0001, name=""):
        self.name = name + "preprocessing(identity)"

    def fit(self, x, *args, **kwargs):
        return self

    def fit_transform(self, x, *args, **kwargs):
        return x

    def transform(self, x, *args, **kwargs):
        return x


class Standardize:
    def __init__(self, eps=0.000001, axis=[0], name=""):
        self.name = name + "preprocessing(standardize,eps=" + str(eps) + ")"
        self.eps = eps
        self.axis = axis

    def fit(self, x, **kwargs):
        print(self.name + " fitting...")
        t = time.time()
        self.mean = x.mean(axis=tuple(self.axis), keepdims=True)
        self.std = x.std(axis=tuple(self.axis), keepdims=True) + self.eps
        print(self.name + " done in {0:.2f} s.".format(time.time() - t))
        return self

    def transform(self, x, inplace=False, **kwargs):
        if inplace:
            x -= self.mean
            x /= self.std
        else:
            return (x - self.mean) / self.std

    def fit_transform(self, x, inplace=False, **kwargs):
        self.fit(x)
        return self.transform(x, inplace)


class ZCAWhitening:
    def __init__(self, eps=0.0001, name=""):
        self.name = name + "preprocessing(zcawhitening,eps=" + str(eps) + ")"
        self.eps = eps

    def fit(self, x):
        print(self.name + " fitting ...")
        t = time.time()
        flatx = np.reshape(x, (x.shape[0], -1))
        self.mean = flatx.mean(0, keepdims=True)
        self.S, self.U = _spectral_decomposition(flatx - self.mean, self.eps)
        print(self.name + " done in {0:.2f} s.".format(time.time() - t))
        return self

    def transform(self, x):
        flatx = np.reshape(x, (x.shape[0], -1)) - self.mean
        return _zca_whitening(flatx, self.U, self.S).reshape(x.shape)

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)


def _spectral_decomposition(flatx, eps):
    U, S, V = np.linalg.svd(flatx, full_matrices=False)
    S = np.diag(1.0 / np.sqrt(S + eps))
    return S, V


def _zca_whitening(flatx, U, S):
    M = np.dot(np.dot(U.T, S), U)
    return np.dot(M, flatx.T).T
