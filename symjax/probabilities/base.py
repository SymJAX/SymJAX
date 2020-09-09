#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

from .. import tensor as T
from .. import nn
import numpy as np

_LOG_2PI = np.log(2 * np.pi)


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


class Normal:
    """(batched, multivariate) normal distribution

    Parameters
    ----------

    mean: N dimensional Tensor
        the mean of the normal distribution, the last
        dimension is the one used to represent the
        dimension of the data, the first dimensions are
        indexed ones

    cov: (N or N+1) dimensional Tensor
        the covariance matrix, if N-dimensional then
        it is assumed to be diagonal, if (N+1)-dimensional
        then the last 2 dimensions are the ones representing
        the covariance dimensions and thus their
        shape should be equal
    """

    def __init__(self, mean, cov):
        if mean.ndim > cov.ndim:
            raise RuntimeError("cov must have at least same rank as mean")

        self.mean = mean
        self.cov = cov
        if mean.ndim == cov.ndim:
            self.log_det_cov = T.log(cov).sum(axis=-1)
            self.prec_U = 1 / T.sqrt(cov)
        else:
            self.prec_U, self.log_det_cov = self._psd_pinv_decomposed_log_pdet(cov)

    def log_prob(self, x):
        """
        Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : Tensor
            samples to use to evaluate the log pdf,
            with the last axis of `x` denoting the components.

        Returns
        -------
        pdf : Tensor
            Log of the probability density function evaluated at `x`
        """
        dim = x.shape[-1]
        dev = x - self.mean
        if self.mean.ndim == self.cov.ndim:
            maha = ((dev * self.prec_U) ** 2).sum(axis=-1)
        else:
            maha = (dev.dot(self.prec_U) ** 2).sum(axis=-1)
        return -0.5 * (dim * _LOG_2PI + self.log_det_cov + maha)

    def prob(self, value):
        """
        Multivariate normal probability density function.

        Parameters
        ----------
        x : Tensor
            samples to use to evaluate the log pdf,
            with the last axis of `x` denoting the components.

        Returns
        -------
        pdf : Tensor
            Probability density function evaluated at `x`
        """
        return T.exp(self.log_prob(value))

    @staticmethod
    def sample(mean, cov):
        """
        Draw random samples from a multivariate normal distribution.

        Parameters
        ----------

        Returns
        -------
        rvs : ndarray or scalar
            Random variates based on given `mean` and `cov`.
        """
        if mean.ndim == cov.ndim:
            return T.random.randn(mean.shape) * T.sqrt(cov) + mean
        else:
            return T.random.randn(mean.shape) * T.linalg.cholesky(cov) + mean

    def entropy(self):
        """
        Compute the differential entropy of the multivariate normal.

        Parameters
        ----------


        Returns
        -------
        h : scalar
            Entropy of the multivariate normal distribution
        """
        if self.mean.ndim == self.cov.ndim:
            return 1 / 2 * T.log(2 * np.pi * np.e * self.cov).sum(-1)
        else:
            return 1 / 2 * T.log(T.linalg.det(2 * np.pi * np.e * self.cov))

    def _psd_pinv_decomposed_log_pdet(mat, cond=1e-5):
        """
        Compute a decomposition of the pseudo-inverse and the logarithm of
        the pseudo-determinant of a symmetric positive semi-definite
        matrix.
        The pseudo-determinant of a matrix is defined as the product of
        the non-zero eigenvalues, and coincides with the usual determinant
        for a full matrix.

        Parameters
        ----------
        mat : array_like
            Input array of shape (`m`, `n`)
        cond: float
            Cutoff for 'small' singular values.
            Eigenvalues smaller than ``cond*largest_eigenvalue``
            are considered zero.
        Returns
        -------
        M : array_like
            The pseudo-inverse of the input matrix is np.dot(M, M.T).
        log_pdet : float
            Logarithm of the pseudo-determinant of the matrix.
        """
        # Compute the symmetric eigendecomposition.
        # The input covariance matrix is required to be real symmetric
        # and positive semidefinite which implies that its eigenvalues
        # are all real and non-negative,
        # but clip them anyway to avoid numerical issues.

        s, u = T.linalg.eigh(mat, lower=True)

        eps = cond * T.max(abs(s))

        s_pinv = T.where(T.abs(s) < eps, 0, 1 / s)

        U = u * T.sqrt(s_pinv)

        log_pdet = T.log(T.where(s > eps, s, 1)).sum()

        return U, log_pdet


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
