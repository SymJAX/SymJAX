#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
from .. import tensor as T
import jax
from ..data.utils import as_tuple
import symjax


__author__ = "Randall Balestriero"

_HERMITE = np.array(
    [[1, 0, -3, 2], [0, 0, 3, -2], [0, 1, -2, 1], [0, 0, -1, 1]],
    dtype="float32",
)


def upsample_1d(
    tensor,
    repeat,
    axis=-1,
    mode="constant",
    value=0.0,
    boundary_condition="periodic",
):
    """1-d upsampling of tensor

    allow to upsample a tensor by an arbitrary (integer) amount on a given
    axis by applying a univariate upsampling strategy.

    Parameters
    ----------

    tensor: tensor
        the input tensor to upsample

    repeat: int
        the amount of new values ot insert between each value

    axis: int
        the axis to upsample

    mode: str
        the type of upsample to perform (linear, constant, nearest)

    value: float (default=0)
        the value ot use for the case of constant upsampling

    """

    if axis == -1:
        axis = tensor.ndim - 1

    if repeat == 0:
        return tensor

    out_shape = list(tensor.shape)
    out_shape[axis] *= 1 + repeat

    if mode == "constant":
        zshape = list(tensor.shape)
        zshape.insert(axis + 1, repeat)
        tensor_aug = T.concatenate(
            [
                T.expand_dims(tensor, axis + 1),
                T.full(zshape, value, dtype=tensor.dtype),
            ],
            axis + 1,
        )

    elif mode == "nearest":
        if boundary_condition == "periodic":
            return T.roll(T.repeat(tensor, repeat + 1, axis), -repeat // 2, axis)
        else:
            raise NotImplemented

    elif mode == "linear":
        assert tensor.shape[axis] > 1
        zshape = [1] * (tensor.ndim + 1)
        zshape[axis + 1] = repeat
        coefficients = T.linspace(0, 1, repeat + 2)[1:-1].reshape(zshape)
        augmented_tensor = T.expand_dims(tensor, axis + 1)
        if boundary_condition == "periodic":
            interpolated = (
                augmented_tensor * (1 - coefficients)
                + T.roll(augmented_tensor, -1, axis) * coefficients
            )
        elif boundary_condition == "mirror":
            assert axis == tensor.ndim - 1
            other = T.index_update(
                T.roll(augmented_tensor, -1, axis),
                T.index[..., -1, :],
                augmented_tensor[..., -2, :],
            )
            interpolated = augmented_tensor * (1 - coefficients) + other * coefficients

        tensor_aug = T.concatenate([augmented_tensor, interpolated], axis + 1)

    return tensor_aug.reshape(out_shape)


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
    mask = ((pos >= 0.0) * (pos <= 1.0)).astype("float32")
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
    patches = T.extract_image_patches(values, (2, 2, 1))  # (n, R_N, R_M, 2, 2, 4)

    F = T.concatenate(
        [
            T.concatenate([patches[..., 0], patches[..., 1]], -1),
            T.concatenate([patches[..., 2], patches[..., 3]], -1),
        ],
        axis=-2,
    )  # (n, R_N, R_M, 4, 4)

    M = T.Variable(
        array([[1, 0, 0, 0], [0, 0, 1, 0], [-3, 3, -2, -1], [2, -2, 1, 1]]).astype(
            "float32"
        ),
        trainable=False,
    )

    MFM = T.einsum("xnmij,ai,bj->xnmab", F, M, M)  # (n, R_N, R_M, 4, 4)

    # filter shape is (n_x,n_y,(n-1)*self.R_N,(n-1)*self.R_M)
    t_x = T.linspace(float32(0), float32(1), int32(n_x - 1))
    t_y = T.linspace(float32(0), float32(1), int32(n_y - 1))
    x = T.pow(t_x, T.arange(4)[:, None])  # (4,T-1)
    y = T.pow(t_y, T.arange(4)[:, None])  # (4,T-1)
    values = T.einsum("xnmij,ia,jb->xnamb", MFM, x, y)
    return T.reshape(values, (n, (R_N) * (n - 1), (R_M) * (n - 1)))


map_coordinates = T.jax_wrap(jax.scipy.ndimage.map_coordinates)


def _meshgrid(height, width):
    # This function is the grid generator from eq. (1) in reference [1].
    # It is equivalent to the following numpy code:
    #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
    #                         np.linspace(-1, 1, height))
    #  ones = np.ones(np.prod(x_t.shape))
    #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    # It is implemented in Theano instead to support symbolic grid sizes.
    # Note: If the image size is known at layer construction time, we could
    # compute the meshgrid offline in numpy instead of doing it dynamically
    # in Theano. However, it hardly affected performance when we tried.
    x_t = T.dot(T.ones((height, 1)), T.linspace(-1.0, 1.0, width)[None])
    y_t = T.dot(T.linspace(-1.0, 1.0, height)[:, None], T.ones((1, width)))

    x_t_flat = x_t.reshape((1, -1))
    y_t_flat = y_t.reshape((1, -1))
    ones = T.ones_like(x_t_flat)
    grid = T.concatenate([x_t_flat, y_t_flat, ones], axis=0)
    return grid


def affine_transform(input, theta, order=1, downsample_factor=1, border_mode="nearest"):
    """
    Spatial transformer layer
    The layer applies an affine transformation on the input. The affine
    transformation is parameterized with six learned parameters [1]_.
    The output is interpolated with a bilinear transformation.
    Parameters
    ----------
    incoming : :class:`Tensor`
        The input which should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
    theta : :class:`Tensor`
        The parameters of the affine
        transformation. See the example for how to initialize to the identity
        transform.
    order: int (default 1)
        The order of the interpolation
    downsample_factor : float or iterable of float
        A float or a 2-element tuple specifying the downsample factor for the
        output image (in both spatial dimensions). A value of 1 will keep the
        original size of the input. Values larger than 1 will downsample the
        input. Values below 1 will upsample the input.
    border_mode : 'nearest', 'mirror', or 'wrap'
        Determines how border conditions are handled during interpolation.  If
        'nearest', points outside the grid are clipped to the boundary. If
        'mirror', points are mirrored across the boundary. If 'wrap', points
        wrap around to the other side of the grid.  See
        http://stackoverflow.com/q/22669252/22670830#22670830 for details.
    References
    ----------
    .. [1]  Max Jaderberg, Karen Simonyan, Andrew Zisserman,
            Koray Kavukcuoglu (2015):
            Spatial Transformer Networks. NIPS 2015,
            http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf
    Examples
    --------
    Here we set up the layer to initially do the identity transform, similarly
    to [1]_. Note that you will want to use a localization with linear output.
    If the output from the localization networks is [t1, t2, t3, t4, t5, t6]
    then t1 and t5 determines zoom, t2 and t4 determines skewness, and t3 and
    t6 move the center position.
    >>> import numpy as np
    >>> import lasagne
    >>> b = np.zeros((2, 3), dtype='float32')
    >>> b[0, 0] = 1
    >>> b[1, 1] = 1
    >>> b = b.flatten()  # identity transform
    >>> W = lasagne.init.Constant(0.0)
    >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
    >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=6, W=W, b=b,
    ... nonlinearity=None)
    >>> l_trans = lasagne.layers.TransformerLayer(l_in, l_loc)
    """

    downsample_factor = as_tuple(downsample_factor, 2)

    input_shp, loc_shp = symjax.current_graph().get([input.shape, theta.shape])

    if len(loc_shp) != 2:
        raise ValueError(
            "The localization network must have " "output shape: (batch_size, 6)"
        )
    if len(input_shp) != 4:
        raise ValueError(
            "The input network must have a 4-dimensional "
            "output shape: (batch_size, num_input_channels, "
            "input_rows, input_columns)"
        )

    return _transform_affine(theta, input, downsample_factor, border_mode, order)


def _transform_affine(theta, input, downsample_factor, border_mode, order):
    num_batch, num_channels, height, width = symjax.current_graph().get(input.shape)
    theta = T.reshape(theta, (-1, 2, 3))

    # grid of (x_t, y_t, 1), eq (1) in ref [1]
    out_height = T.cast(height // downsample_factor[0], "int64")
    out_width = T.cast(width // downsample_factor[1], "int64")
    grid = _meshgrid(out_height, out_width)

    # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
    transformed_points = T.dot(theta, grid)

    transformed_points = (transformed_points + 1) / 2
    transformed_points = transformed_points * np.array([[width], [height]])
    output = T.map(
        lambda a, b: T.stack(
            [
                T.interpolation.map_coordinates(a[i], b[::-1], order=order)
                for i in range(num_channels)
            ]
        ),
        sequences=[input, transformed_points],
    )
    output = output.reshape((num_batch, num_channels, out_height, out_width))
    return output


def thin_plate_spline(
    input, dest_offsets, order=1, downsample_factor=1, border_mode="nearest"
):
    """
    applies a thin plate spline transformation [2]_ on the input
    as in [1]_.

    The thin plate spline transform is determined based on the
    movement of some number of control points. The starting positions for
    these control points are fixed. The output is interpolated with a
    bilinear transformation.

    Implementation based on Lasagne

    Parameters
    ----------
    incoming : Tensor
        The input to be transformed, should be a 4D tensor, with shape
        ``(batch_size, num_input_channels, input_rows, input_columns)``.
    dest_offsets : Tensor
        The parameters of the thin plate spline
        transformation as the x and y coordinates of the destination offsets of
        each control point. This should be a
        2D tensor, with shape ``(batch_size, 2 * num_control_points)``.
        The number of control points to be used for the thin plate spline
        transformation. These points will be arranged as a grid along the
        image, so the value must be a perfect square. Default is 16.
    order: int (default 1)
        The order of the interpolation
    downsample_factor : float or iterable of float
        A float or a 2-element tuple specifying the downsample factor for the
        output image (in both spatial dimensions). A value of 1 will keep the
        original size of the input. Values larger than 1 will downsample the
        input. Values below 1 will upsample the input.
    border_mode : 'nearest', 'mirror', or 'wrap'
        Determines how border conditions are handled during interpolation.  If
        'nearest', points outside the grid are clipped to the boundary'. If
        'mirror', points are mirrored across the boundary. If 'wrap', points
        wrap around to the other side of the grid.  See
        http://stackoverflow.com/q/22669252/22670830#22670830 for details.
    References
    ----------
    .. [1]  Max Jaderberg, Karen Simonyan, Andrew Zisserman,
            Koray Kavukcuoglu (2015):
            Spatial Transformer Networks. NIPS 2015,
            http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf
    .. [2]  Fred L. Bookstein (1989):
            Principal warps: thin-plate splines and the decomposition of
            deformations. IEEE Transactions on
            Pattern Analysis and Machine Intelligence.
            http://doi.org/10.1109/34.24792
    """

    downsample_factor = as_tuple(downsample_factor, 2)

    input_shp, loc_shp = symjax.current_graph().get([input.shape, dest_offsets.shape])
    control_points = loc_shp[1] // 2
    control_points = control_points

    # Error checking
    if len(loc_shp) != 2:
        raise ValueError(
            "The localization network must have "
            "output shape: (batch_size, "
            "2*control_points)"
        )

    if len(input_shp) != 4:
        raise ValueError(
            "The input network must have a 4-dimensional "
            "output shape: (batch_size, num_input_channels, "
            "input_rows, input_columns)"
        )

    # Create source points and L matrix
    (
        right_mat,
        L_inv,
        source_points,
        out_height,
        out_width,
    ) = _initialize_tps(control_points, input_shp, downsample_factor)

    # compute output
    # see eq. (1) and sec 3.1 in [1]
    # Get input and destination control points
    return _transform_thin_plate_spline(
        dest_offsets,
        input,
        right_mat,
        L_inv,
        source_points,
        out_height,
        out_width,
        downsample_factor,
        border_mode,
        order,
    )


def _initialize_tps(num_control_points, input_shape, downsample_factor):
    """
    Initializes the thin plate spline calculation by creating the source
    point array and the inverted L matrix used for calculating the
    transformations as in ref [2]_
    :param num_control_points: the number of control points. Must be a
        perfect square. Points will be used to generate an evenly spaced grid.
    :param input_shape: tuple with 4 elements specifying the input shape
    :param downsample_factor: tuple with 2 elements specifying the
        downsample for the height and width, respectively
    :param precompute_grid: boolean specifying whether to precompute the
        grid matrix
    :return:
        right_mat: shape (num_control_points + 3, out_height*out_width) tensor
        L_inv: shape (num_control_points + 3, num_control_points + 3) tensor
        source_points: shape (2, num_control_points) tensor
        out_height: tensor constant specifying the ouptut height
        out_width: tensor constant specifying the output width
    """

    # break out input_shape
    _, _, height, width = input_shape

    # Create source grid
    grid_size = np.sqrt(num_control_points)
    x_control_source, y_control_source = np.meshgrid(
        np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size)
    )

    # Create 2 x num_points array of source points
    source_points = np.vstack((x_control_source.flatten(), y_control_source.flatten()))

    # Convert to floatX
    source_points = source_points.astype("float32")

    # Get number of equations
    num_equations = num_control_points + 3

    # Initialize L to be num_equations square matrix
    L = np.zeros((num_equations, num_equations), dtype="float32")

    # Create P matrix components
    L[0, 3:num_equations] = 1.0
    L[1:3, 3:num_equations] = source_points
    L[3:num_equations, 0] = 1.0
    L[3:num_equations, 1:3] = source_points.T

    # Loop through each pair of points and create the K matrix
    for point_1 in range(num_control_points):
        for point_2 in range(point_1, num_control_points):

            L[point_1 + 3, point_2 + 3] = _U_func_numpy(
                source_points[0, point_1],
                source_points[1, point_1],
                source_points[0, point_2],
                source_points[1, point_2],
            )

            if point_1 != point_2:
                L[point_2 + 3, point_1 + 3] = L[point_1 + 3, point_2 + 3]

    # Invert
    L_inv = np.linalg.inv(L)

    # Construct grid
    out_height = np.array(height // downsample_factor[0]).astype("int64")
    out_width = np.array(width // downsample_factor[1]).astype("int64")
    x_t, y_t = np.meshgrid(
        np.linspace(-1, 1, out_width), np.linspace(-1, 1, out_height)
    )
    ones = np.ones(np.prod(x_t.shape))
    orig_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    orig_grid = orig_grid[0:2, :]
    orig_grid = orig_grid.astype("float32")

    # Construct right mat

    # First Calculate the U function for the new point and each source
    # point as in ref [2]
    # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
    # squared distance
    to_transform = orig_grid[:, :, np.newaxis].transpose(2, 0, 1)
    stacked_transform = np.tile(to_transform, (num_control_points, 1, 1))
    stacked_source_points = source_points[:, :, np.newaxis].transpose(1, 0, 2)
    r_2 = np.sum((stacked_transform - stacked_source_points) ** 2, axis=1)

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
    log_r_2 = np.log(r_2)
    log_r_2[np.isinf(log_r_2)] = 0.0
    distances = r_2 * log_r_2

    # Add in the coefficients for the affine translation (1, x, and y,
    # corresponding to a_1, a_x, and a_y)
    upper_array = np.ones(shape=(1, orig_grid.shape[1]), dtype="float32")
    upper_array = np.concatenate([upper_array, orig_grid], axis=0)
    right_mat = np.concatenate([upper_array, distances], axis=0)

    return right_mat, L_inv, source_points, out_height, out_width


def _U_func_numpy(x1, y1, x2, y2):
    """
    Function which implements the U function from Bookstein paper
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: value of z
    """

    # Return zero if same point
    if x1 == x2 and y1 == y2:
        return 0.0

    # Calculate the squared Euclidean norm (r^2)
    r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Return the squared norm (r^2 * log r^2)
    return r_2 * np.log(r_2)


def _transform_thin_plate_spline(
    dest_offsets,
    input,
    right_mat,
    L_inv,
    source_points,
    out_height,
    out_width,
    downsample_factor,
    border_mode,
    order,
):

    num_batch, num_channels, height, width = symjax.current_graph().get(input.shape)
    num_control_points = source_points.shape[1]

    # reshape destination offsets to be (num_batch, 2, num_control_points)
    # and add to source_points
    dest_points = source_points + T.reshape(
        dest_offsets, (num_batch, 2, num_control_points)
    )

    # Solve as in ref [2]
    coefficients = T.dot(dest_points, L_inv[:, 3:].T)

    # Transform each point on the source grid (image_size x image_size)
    right_mat = T.tile(right_mat[None], (num_batch, 1, 1))
    transformed_points = T.einsum("abc,acd->abd", coefficients, right_mat)
    transformed_points = (transformed_points + 1) / 2
    transformed_points = transformed_points * np.array([[width], [height]])
    output = T.map(
        lambda a, b: T.stack(
            [
                T.interpolation.map_coordinates(a[i], b[::-1], order=order)
                for i in range(num_channels)
            ]
        ),
        sequences=[input, transformed_points],
    )

    output = T.reshape(output, (num_batch, num_channels, out_height, out_width))

    return output


def _get_transformed_points_tps(
    new_points, source_points, coefficients, num_points, batch_size
):
    """
    Calculates the transformed points' value using the provided coefficients
    :param new_points: num_batch x 2 x num_to_transform tensor
    :param source_points: 2 x num_points array of source points
    :param coefficients: coefficients (should be shape (num_batch, 2,
        control_points + 3))
    :param num_points: the number of points
    :return: the x and y coordinates of each transformed point. Shape (
        num_batch, 2, num_to_transform)
    """

    # Calculate the U function for the new point and each source point as in
    # ref [2]
    # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
    # squared distance

    # Calculate the squared dist between the new point and the source points
    to_transform = new_points[:, None]
    stacked_transform = T.tile(to_transform, (1, num_points, 1, 1))
    r_2 = T.sum(
        ((stacked_transform - source_points.transpose()[None, :, :, None]) ** 2),
        axis=2,
    )

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
    log_r_2 = T.log(r_2)
    distances = T.where(T.isnan(log_r_2), r_2 * log_r_2, 0.0)

    # Add in the coefficients for the affine translation (1, x, and y,
    # corresponding to a_1, a_x, and a_y)
    upper_array = T.concatenate(
        [
            T.ones(
                (batch_size, 1, new_points.shape[2]),
            ),
            new_points,
        ],
        axis=1,
    )
    right_mat = T.concatenate([upper_array, distances], axis=1)

    # Calculate the new value as the dot product
    new_value = T.batched_dot(coefficients, right_mat)
    return new_value
