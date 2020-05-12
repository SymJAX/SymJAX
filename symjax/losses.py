from . import tensor as T
import numpy as np

def vae(x, x_hat, z_mu, z_logvar, mu, logvar, logvar_x=0., eps=1e-8):
    """N samples of dimension D to latent space in K dimension

    Parameters
    ----------

    x: array
        should be of shape (N, D)

    x_hat: array
        should be of shape (N, D)

    z_mu: array
        should be of shape (N, K), infered mean of variational Gaussian

    z_logvar: array
        should be of shape (N, K), infered log-variance of variational Gaussian

    mu: array
        should be of shape (K,), parameter (centroids)

    logvar: array
        should be of shape (K,), parameter (logvar of clusters)

    """

    var = T.exp(logvar)

    # E_{q(z,c|x)}[log(p(x|z))]
    px_z = - 0.5 * ((x - x_hat) ** 2 / T.exp(logvar_x) + logvar_x).sum(1)

    # - E_{q(z,c|x)}[log(q(z|x))] : entropy of normal
    h_z = 0.5 * z_logvar.sum(1)

    # E_{q(z,c|x)}[log(p(z|c)]
    ll_z = - 0.5 * (logvar + z_var / var - 1 + (z_mu - mu) ** 2 / var).sum(-1)

    loss = - (px_z + ll_z + h_z)

    return loss



def vae_gmm(x, x_hat, z_mu, z_logvar, mu, logvar, logpi, logvar_x=0., eps=1e-8):
    """N samples of dimension D to latent space of C sluters in K dimension

    Parameters
    ----------

    x: array
        should be of shape (N, D)

    x_hat: array
        should be of shape (N, D)

    z_mu: array
        should be of shape (N, K), infered mean of variational Gaussian

    z_logvar: array
        should be of shape (N, K), infered log-variance of variational Gaussian

    mu: array
        should be of shape (C, K), parameter (centroids)

    logvar: array
        should be of shape (C, K), parameter (logvar of clusters)

    logpi: array
        should be of shape (C,), parameter (prior of clusters)

    """

    var = T.exp(logvar)
    z_var = T.exp(z_logvar)

    # predict the log probability of clusters, shape will be (N, C)
    # and compute compute p(t|z) = p(z|t)p(t)/(\sum_t p(z|t)p(t))
    logprob = (logpi[:, None] - .5 * (T.log(2 * np.pi) + logvar)\
                    - (z_mu[:, None, :] - mu) ** 2 / (2 * var)).sum(2)
    pt_z = T.softmax(logprob)

    # E_{q(z,c|x)}[log(p(x|z))]
    px_z = - 0.5 * ((x - x_hat) ** 2 / T.exp(logvar_x)+ logvar_x ).sum(1)

    # - E_{q(z,c|x)}[log(q(c|x))] entropy of categorical
    h_c = - (pt_z * T.log_softmax(logprob)).sum(1)

    # - E_{q(z,c|x)}[log(q(z|x))] : entropy of normal
    h_z = 0.5 * z_logvar.sum(1)

    # E_{q(z,c|x)}[log(p(z|c)]
    ll_z = - 0.5 * (pt_z * (logvar + z_var[:, None, :] / var - 1\
            + (z_mu[:, None, :] - mu) ** 2 / var).sum(-1)).sum(-1)

    # E_{q(z,c|x)}[log(p(c)]
    p_c = (pt_z * logpi).sum(1)

    loss = -(px_z + ll_z + p_c + h_c + h_z)

    return loss, px_z + p_c, pt_z

 
def vae_comp_gmm(x, x_hat, z_mu, z_logvar, mu, logvar, logpi, logvar_x=0., eps=1e-8):
    """N samples of dimension D to latent space of I pieces each of C sluters
    in K dimension

    Parameters
    ----------

    x: array
        should be of shape (N, D)

    x_hat: array
        should be of shape (N, D)

    z_mu: array
        should be of shape (N, I, K), infered mean of variational Gaussian

    z_logvar: array
        should be of shape (N, I, K), infered log-variance of variational Gaussian

    mu: array
        should be of shape (I, C, K), parameter (centroids)

    logvar: array
        should be of shape (I, C, K), parameter (logvar of clusters)

    logpi: array
        should be of shape (I, C), parameter (prior of clusters)

    """

    var = T.exp(logvar)

    # predict the log probability of clusters, shape will be (N, I, C)
    # and compute compute p(t_i|z_i) = p(z_i|t_i)p(t_i)/(\sum_t_i p(z_i|t_i)p(t_i))
    logprob = (logpi[:, :, None] - .5 * (T.log(2 * np.pi) + logvar)\
                - (z_mu[:, :, None, :] - mu) ** 2 / (2 * var)).sum(3)
    pt_z = T.softmax(logprob)

    # E_{q(z,c|x)}[log(p(x|z))]
    px_z = ((x - x_hat)**2).sum(1)

    # - E_{q(z,c|x)}[log(q(c|x))] entropy of categorical
    h_c = - (pt_z * T.log_softmax(logprob)).sum((1, 2))

    # - E_{q(z,c|x)}[log(q(z|x))] : entropy of normal
    h_z = 0.5 * z_logvar.sum((1, 2))

    # E_{q(z,c|x)}[log(p(z|c)]
    ll_z = - 0.5 * (pt_z * (logvar + z_var[:, :, None, :] / var - 1\
            + (z_mu[:, :, None, :] - mu) ** 2 / var).sum(-1)).sum((1, 2))

    # E_{q(z,c|x)}[log(p(c)]
    p_c = (pt_z * logpi[:, :, None]).sum((1, 2))

    loss = - (px_z + ll_z + p_c + h_c + h_z)

    return loss

  
def vae_fact_gmm(x, x_hat, z_mu, z_logvar, mu, logvar, logpi, eps=1e-8):
    """N samples of dimension D to latent space of I pieces each of C sluters
    in K dimension

    Parameters
    ----------

    x: array
        should be of shape (N, D)

    x_hat: array
        should be of shape (N, D)

    z_mu: array
        should be of shape (N, K), infered mean of variational Gaussian

    z_logvar: array
        should be of shape (N, K), infered log-variance of variational Gaussian

    mu: array
        should be of shape (I, C, K), parameter (centroids)

    logvar: array
        should be of shape (K,), parameter (logvar of clusters)

    logpi: array
        should be of shape (I, C), parameter (prior of clusters)

    """

    var = T.exp(logvar)

    # predict the log probability of clusters, shape will be (N, I, C)
    # and compute compute p(t_i|z_i) = p(z_i|t_i)p(t_i)/(\sum_t_i p(z_i|t_i)p(t_i))
    logprob = (logpi[:, :, None] - .5 * (T.log(2 * np.pi) + logvar)\
                - (z_mu[:, :, None, :] - mu) ** 2 / (2 * var)).sum(3)
    pt_z = T.softmax(logprob)

    # E_{q(z,c|x)}[log(p(x|z))]
    px_z = ((x - x_hat)**2).sum(1)

    # - E_{q(z,c|x)}[log(q(c|x))] entropy of categorical
    h_c = - (pt_z * T.log_softmax(logprob)).sum((1, 2))

    # - E_{q(z,c|x)}[log(q(z|x))] : entropy of normal
    h_z = 0.5 * z_logvar.sum((1, 2))

    # E_{q(z,c|x)}[log(p(z|c)]
    ll_z = - 0.5 * (logvar + z_var / var - 1\
            + ((z_mu ** 2 - 2 * z_mu * (mu * pt_z[...,None]).sum((1, 2))\
                + (mu * pt_z[...,None]).sum((1, 2)) ** 2\
                - (mu* pt_z[...,None]**2).sum((1, 2))\
                + (mu* pt_z[...,None]).sum((1, 2)))/ var).sum(-1))

    # E_{q(z,c|x)}[log(p(c)]
    p_c = (pt_z * logpi[:, :, None]).sum((1, 2))

    loss = - (px_z + ll_z + p_c + h_c + h_z)

    return loss





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
