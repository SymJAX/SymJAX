#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

from .. import tensor as T
from .. import nn
from ..tensor.base import jax_wrap
import numpy as np
import jax


class Categorical:
    def __init__(self, probabilities=None, logits=None):
        if logits is not None:
            self.logits = logits
        else:
            assert probabilities is not None
            self.logits = T.log(probabilities)

        self.log_probabilities = nn.log_softmax(logits)

    def log_prob(self, value):
        return T.take_along_axis(self.log_probabilities, value[:, None], 1).squeeze(1)

    def prob(self, value):
        return T.exp(self.log_prob(value))

    def sample(self):
        return T.random.categorical(self.logits)

    def entropy(self):
        return -T.sum(T.exp(self.log_probabilities) * self.log_probabilities, -1)


class MultivariateNormal:
    def __init__(self, mean, diag_std=None, diag_log_std=None):
        self.mean = mean
        if diag_log_std is not None:
            self.diag_log_std = diag_log_std
        else:
            assert diag_std is not None
            self.diag_log_std = T.log(diag_std)

    def log_prob(self, value):
        pre_sum = -0.5 * (
            ((x - mu) / tf.exp(log_std)) ** 2 + 2 * log_std + np.log(2 * np.pi)
        )
        return pre_sum.sum(-1)

    def prob(self, value):
        return T.exp(self.log_prob(value))

    def sample(self):
        return T.random.randn(self.mean.shape) * T.exp(self.diag_log_std) + self.mean

    def entropy(self):
        cst = T.log(2 * np.pi) + 1
        return self.diag_log_std.sum() + self.diag_log_std.shape[-1] * cst


def KL(X, Y):
    """
    Normal:
    distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)
    """
    if isinstance(X, MultivariateNormal) and isinstance(Y, MultivariateNormal):
        mu_0 = X.mean
        mu_1 = Y.mean
        log_std0 = X.diag_log_std
        log_std1 = Y.diag_log_std

        var0, var1 = tf.exp(2 * log_std0), tf.exp(2 * log_std1)
        pre_sum = (
            0.5 * (((mu1 - mu0) ** 2 + var0) / (var1 + EPS) - 1) + log_std1 - log_std0
        )
        all_kls = pre_sum.sum(-1)
        return all_kls

    elif isinstance(X, Categorical) and isinstance(Y, Categorical):
        all_kls = (
            T.exp(Y.log_probabilities) * (T.log_probabilities - X.log_probabilities)
        ).sum(-1)
        return all_kls
