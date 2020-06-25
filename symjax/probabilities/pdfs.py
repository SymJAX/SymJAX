#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

from .. import tensor
import numpy as np


_LOG_2PI = np.log(2 * np.pi)
_LOG_2 = np.log(2)
_LOG_PI = np.log(np.pi)


def psd(M):
    s, u = tensor.linalg.eigh(M, "L")
    # to do this function
    # s_pinv = np.array([0 if abs(x) <= eps else 1/x for x in v],
    #                    dtype=float)
    U = u * tensor.sqrt(1 / s)
    return U, tensor.log(s).sum()


class multivariate_normal:
    def logpdf(x, mean, cov):
        """
        Log of the multivariate normal probability density function.

        Parameters
        ----------
        x : array_like

        Returns
        -------
        pdf : ndarray
            Log of the probability density function evaluated at `x`
            :param mean:
            :param cov:

        """

        # we get the precision matrix such that
        # precision = dot(prec_U, prec_U.T)
        prec_U, log_det = psd(cov)
        centered_x = x - mean
        log_exp = (tensor.dot(centered_x, prec_U) ** 2).sum(axis=-1)
        return -0.5 * (x.shape[-1] * _LOG_2PI + log_det + log_exp)

        return _squeeze_output(out)

    def pdf(x, mean, cov):
        """
        Multivariate normal probability density function.
        Parameters
        ----------
        x : array_like
            Quantiles, with the last axis of `x` denoting the components.
        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at `x`
            :param mean:
            :param cov:
        """
        return tensor.exp(multivariate_normal.logpdf(x, mean, cov))
