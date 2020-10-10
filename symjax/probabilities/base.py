#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

from .. import tensor as T
from .. import nn
import numpy as np

_LOG_2PI = np.log(2 * np.pi)


class Distribution:
    def prob(self, value):
        pass

    def log_prob(self, value):
        pass

    def sample(self, value):
        pass

    def entropy(self, value):
        pass


class Categorical(Distribution):
    def __init__(self, probabilities=None, logits=None, eps=1e-8):
        if logits is not None:
            self.log_probabilities = nn.log_softmax(logits)
            self.probabilities = T.exp(self.log_probabilities)
        else:
            assert probabilities is not None
            self.log_probabilities = T.log(probabilities + eps)
            self.probabilities = probabilities

    def log_prob(self, value):
        # case where the values are discrete
        return T.take_along_axis(self.log_probabilities, value[..., None], -1).squeeze(
            -1
        )

    def prob(self, value):
        return T.exp(self.log_prob(value))

    def sample(self):
        return T.random.categorical(self.log_probabilities)

    def entropy(self):
        return -T.sum(self.probabilities * self.log_probabilities, -1)


class Normal(Distribution):
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
            self.diagonal_cov = True
        else:
            self.prec_U, self.log_det_cov = self._psd_pinv_decomposed_log_pdet(cov)
            self.diagonal_cov = False

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

    def sample(self):
        """
        Draw random samples from a multivariate normal distribution.

        Parameters
        ----------

        Returns
        -------
        rvs : ndarray or scalar
            Random variates based on given `mean` and `cov`.
        """
        if self.mean.ndim == self.cov.ndim:
            return T.random.randn(self.mean.shape) * T.sqrt(self.cov) + self.mean
        else:
            return (
                T.random.randn(self.mean.shape) * T.linalg.cholesky(self.cov)
                + self.mean
            )

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
            # we do not need the signs since it is positive
            return 1 / 2 * T.linalg.slogdet(2 * np.pi * np.e * self.cov)[1]

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


def cross_entropy(X, Y):
    if isinstance(X, Categorical) and isinstance(Y, Categorical):
        return -(X.probabilities * Y.log_probabilities).sum(-1)


def KL(X, Y, EPS=1e-8):
    """
    Normal:
    distributions are specified by means and log stds.
    (https://en.wikipedia.org/wiki/Kullback-Leibler_divergence#Multivariate_normal_distributions)

    .. math::

        KL(p||q)=\\int [\\log(p(x))-\\log(q(x))]p(x)dx

        =\\int[\\frac{1}{2}log(\\frac{|\\Sigma_2|}{|\\Sigma_1|})−\\frac{1}{2}(x−\\mu_1)^𝑇\\Sigma_1^{-1}(x−\\mu_1)+\\frac{1}{2}(x−\\mu_2)^𝑇\\Sigma_2^{−1}(x−\\mu_2)] p(x)dx

        =\\frac{1}{2}log(\\frac{|\\Sigma_2|}{|\\Sigma_1|})−\\frac{1}{2}tr {𝐸[(x−\\mu_1)(x−\\mu_1)^𝑇] Σ−11}+\\frac{1}{2}𝐸[(x−\\mu_2)^𝑇\\Sigma_2^{−1}(x−\\mu_2)]

        =\\frac{1}{2}log(\\frac{|\\Sigma_2|}{|\\Sigma_1|})−\\frac{1}{2}tr {𝐼𝑑}+\\frac{1}{2}(\\mu_1−\\mu_2)^𝑇Σ_2^{-1}(\\mu_1−\\mu_2)+\\frac{1}{2}tr{\\Sigma_2^{-1}\\Sigma_1}

        =\\frac{1}{2}[log(\\frac{|\\Sigma_2|}{|\\Sigma_1|})−𝑑+tr{\\Sigma_2^{−1}\\Sigma_1}+(\\mu_2−\\mu_1)^𝑇\\Sigma_2^{−1}(\\mu_2−\\mu_1)].

    """
    if isinstance(X, Normal) and isinstance(Y, Normal):
        mu0 = X.mean
        mu1 = Y.mean
        var0 = X.cov
        var1 = Y.cov

        if X.diagonal_cov and Y.diagonal_cov:
            pre_sum = (((mu1 - mu0) ** 2 + var0) / (var1 + EPS) - 1) + T.log(
                var1 / (var0 + EPS)
            )

            return 1 / 2 * pre_sum.sum(-1)

    elif isinstance(X, Categorical) and isinstance(Y, Categorical):
        all_kls = (
            T.exp(Y.log_probabilities) * (T.log_probabilities - X.log_probabilities)
        ).sum(-1)
        return all_kls
