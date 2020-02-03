from . import tensor as T


def sparse_crossentropy_logits(p, q, weights=None):
    """Cross entropy loss given that :math:`p` is sparse and
    :math:`q` is the log-probability.

    The formal definition given that :math:`p` is now an
    index (of the Dirac) s.a. :math:`p\in \{1,\dots,D\}`
    and :math:`q` is unormalized (log-proba)
    is given by (for discrete variables, p sparse)

    .. math::
        \mathcal{L}(p,q)=-q_{p}+\log(\sum_{d=1}^D \exp(q_d))
    .. math::
        \mathcal{L}(p,q)=-q_{p}+LogSumExp(q)
    .. math::
        \mathcal{L}(p,q)=-q_{p}+LogSumExp(q-\max_{d}q_d)

    or by (non p sparse)

    .. math::
        \mathcal{L}(p,q)=-\sum_{d=1}^Dp_{d}q_{d}+\log(\sum_{d=1}^D \exp(q_d))
    .. math::
        \mathcal{L}(p,q)=-\sum_{d=1}^Dp_{d}q_{d}+LogSumExp(q)
    .. math::
        \mathcal{L}(p,q)=-\sum_{d=1}^Dp_{d}q_{d}+LogSumExp(q-\max_{d}q_d)


    with :math:`p` the class index and :math:`q` the predicted one
    (output of the network). This class takes two non sparse
    vectors which should be nonnegative and sum to one.
    """
    # the linear part of the loss
    linear = T.take_along_axis(q, p[:, None], 1).squeeze()
    logsumexp = T.logsumexp(q, 1)
    if weights is not None:
        return weights * (-linear + logsumexp)
    else:
        return logsumexp - linear


def crossentropy_logits(p, q, p_sparse=True):
    """see sparse cross entropy"""
    linear = (p * q).sum(1)
    logsumexp = T.logsumexp(q, 1)
    if weights is not None:
        return weights * (-linear + logsumexp)
    else:
        return -linear + logsumexp


def sigmoid_crossentropy_logits(labels, logits):
    return - logits * labels + T.log1p(T.exp(logits))



def multiclass_hinge_loss(predictions, targets, delta=1):
    """Computes the multi-class hinge loss between predictions and targets.
    .. math:: L_i = \\max_{j \\not = t_i} (0, p_j - p_{t_i} + \\delta)
    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of one-hot encoding of the correct class in the same
        layout as predictions (non-binary targets in [0, 1] do not work!)
    delta : scalar, default 1
        The hinge loss margin
    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise multi-class hinge loss
    Notes
    -----
    This is an alternative to the categorical cross-entropy loss for
    multi-class classification problems
    """
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = T.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')
    corrects = predictions[targets.nonzero()]
    rest = predictions[(1 - targets).nonzero()].reshape((-1, num_cls - 1))
    rest = rest.max(axis=1)
    return T.activations.relu(rest - corrects + delta)


def squared_difference(targets, predictions):
    return (targets - predictions)**2


def accuracy(targets, predictions):
    if predictions.ndim == 2:
        accu = T.cast(T.equal(targets, predictions.argmax(1)), 'float32')
    else:
        accu = T.cast(T.equal(targets, predictions), 'float32')
    return accu.mean()
