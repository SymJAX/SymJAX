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
    linear = T.take(q, p, 1)
    # LogSumExp with max removal
    q_max = T.stop_gradient(q.max(1, keepdims=True))
    logsumexp = T.log(T.exp(q-q_max).sum(1))+q_max[:,0]
    if weights is not None:
        return T.mean(weights*(-linear + logsumexp))
    else:
        return T.mean(-linear + logsumexp)


def crossentropy_logits(p, q, weights=None, p_sparse=True):
    """see sparse cross entropy"""
    linear = (p*q).sum(1)
    # LogSumExp with max removal
    q_max = T.stop_gradient(q.max(1, keepdims=True))
    logsumexp = T.log(T.exp(q-q_max).sum(1))+q_max[:,0]
    if weights is not None:
        return T.mean(weights*(-linear + logsumexp))
    else:
        return T.mean(-linear + logsumexp)



def accuracy(targets, predictions):
    if predictions.ndim == 2:
        predictions = predictions.argmax(1)
    accu = T.cast(T.equal(targets, predictions), 'float32').mean()
    return accu


