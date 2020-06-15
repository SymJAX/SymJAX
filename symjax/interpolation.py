#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

import symjax.tensor as T

__author__ = "Randall Balestriero"

_HERMITE = np.array([[1, 0, -3, 2],
                     [0, 0, 3, -2],
                     [0, 1, -2, 1],
                     [0, 0, -1, 1]], dtype='float32')


def hermite_1d(samples, knots, values, derivatives):
    """Real interpolation with hermite cubic spline.

    Arguments
    ---------
        knots: array-like
            The knots onto the function is defined (derivative and
            antiderivative) tensor of knots can be given in which case
            the shape is (..., N_KNOTS) the first dimensions are treated
            independently.
        samples: array-like
            The points where the interpolation is required of shape
            (TIME). If the shape is more, the first dimensions must be
            broadcastable agains knots.
        values: array-like
            The real values of amplitude onto knots, same shape as knots.
        derivative: array-like
            The real values of derivatives onto knots, same shape as knots.

    Returns
    -------
        yi: array-like
            The interpolated real-valued function.
            :param derivatives:
    """

    # Concatenate coefficients onto shifted knots (..., N_KNOTS - 1, 2)
    adj_knots = T.stack([knots[..., :-1], knots[..., 1:]], axis=-1)
    adj_v = T.stack([values[..., :-1], values[..., 1:]], axis=-1)
    adj_d = T.stack([derivatives[..., :-1], derivatives[..., 1:]], axis=-1)

    # Define the full function y to interpolate (..., N_KNOTS - 1, 4)
    adj_vd = T.concatenate([adj_v, adj_d], axis=-1)

    # Extract Hermite polynomial coefficients (..., N_KNOTS - 1, 4)
    yh = T.matmul(adj_vd, _HERMITE)

    # Now we must apply a duplication over the number of knots, apply the
    # polynomial interpolation, and then mask for each region and sum
    # over the regions (..., N_KNOTS - 1, TIME)
    if samples.ndim == 1:
        samples_ = samples.reshape([1] * knots.ndim + [-1])
    else:
        samples_ = T.expand_dims(samples, -2)

    # we keep left and right and they will coincide for adjacent regions
    start = T.expand_dims(knots[..., :-1], -1)
    end = T.expand_dims(knots[..., 1:], -1)
    pos = (samples_ - start) / (end - start)
    mask = ((pos >= 0.) * (pos <= 1.0)).astype('float32')
    mask = mask / T.maximum(1, mask.sum(-2, keepdims=True))

    # create the polynomial basis (..., N_KNOTS - 1, TIME, 4)
    polynome = T.expand_dims(pos, -1) ** T.arange(4)

    # apply mask
    mask_polynome = polynome * T.expand_dims(mask, -1)

    # linearly combine to produce interpolation
    return (T.expand_dims(yh, -2) * mask_polynome).sum(axis=(-3, -1))


def hermite_2d(values, n_x, n_y):
    """
    TODO: test and finalize this

    Parameters
    ----------

    values: array-like
        the values, and 2 directional derivatives and the cross derivative
        for the 4 knots per region, hence it should be of shape n,N,M,4
        values vx vy vxy

    n_x: int
        the number of points in x per region

    n_y: int
        the number of points in y per region

    Returns
    -------

    interpolation: array-like

    """
    n, N, M = values.shape[:3]
    R_N = N - 1
    R_M = M - 1
    patches = T.extract_image_patches(values,
                                      (2, 2, 1))  # (n, R_N, R_M, 2, 2, 4)

    F = T.concatenate([T.concatenate([patches[..., 0], patches[..., 1]], -1),
                       T.concatenate([patches[..., 2], patches[..., 3]], -1)],
                      axis=-2)  # (n, R_N, R_M, 4, 4)

    M = T.Variable(array([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [-3, 3, -2, -1],
                          [2, -2, 1, 1]]).astype('float32'), trainable=False)

    MFM = T.einsum('xnmij,ai,bj->xnmab', F, M, M)  # (n, R_N, R_M, 4, 4)

    # filter shape is (n_x,n_y,(n-1)*self.R_N,(n-1)*self.R_M)
    t_x = T.linspace(float32(0), float32(1), int32(n_x - 1))
    t_y = T.linspace(float32(0), float32(1), int32(n_y - 1))
    x = T.pow(t_x, T.arange(4)[:, None])  # (4,T-1)
    y = T.pow(t_y, T.arange(4)[:, None])  # (4,T-1)
    values = T.einsum('xnmij,ia,jb->xnamb', MFM, x, y)
    return T.reshape(values, (n, (R_N) * (n - 1), (R_M) * (n - 1)))
