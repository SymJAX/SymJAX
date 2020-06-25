from .. import tensor as T
import numpy as np


def vae(x, x_hat, z_mu, z_logvar, mu, logvar, logvar_x=0.0, eps=1e-8):
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
        :param logvar_x:
        :param eps:

    """

    var = T.exp(logvar)

    # E_{q(z,c|x)}[log(p(x|z))]
    px_z = -0.5 * ((x - x_hat) ** 2 / T.exp(logvar_x) + logvar_x).sum(1)

    # - E_{q(z,c|x)}[log(q(z|x))] : entropy of normal
    h_z = 0.5 * z_logvar.sum(1)

    # E_{q(z,c|x)}[log(p(z|c)]
    ll_z = -0.5 * (logvar + z_var / var - 1 + (z_mu - mu) ** 2 / var).sum(-1)

    loss = -(px_z + ll_z + h_z)

    return loss


def vae_gmm(x, x_hat, z_mu, z_logvar, mu, logvar, logpi, logvar_x=0.0, eps=1e-8):
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
        :param logvar_x:
        :param eps:

    """

    var = T.exp(logvar)
    z_var = T.exp(z_logvar)

    # predict the log probability of clusters, shape will be (N, C)
    # and compute compute p(t|z) = p(z|t)p(t)/(\sum_t p(z|t)p(t))
    logprob = (
        logpi[:, None]
        - 0.5 * (T.log(2 * np.pi) + logvar)
        - (z_mu[:, None, :] - mu) ** 2 / (2 * var)
    ).sum(2)
    pt_z = T.softmax(logprob)

    # E_{q(z,c|x)}[log(p(x|z))]
    px_z = -0.5 * ((x - x_hat) ** 2 / T.exp(logvar_x) + logvar_x).sum(1)

    # - E_{q(z,c|x)}[log(q(c|x))] entropy of categorical
    h_c = -(pt_z * T.log_softmax(logprob)).sum(1)

    # - E_{q(z,c|x)}[log(q(z|x))] : entropy of normal
    h_z = 0.5 * z_logvar.sum(1)

    # E_{q(z,c|x)}[log(p(z|c)]
    ll_z = -0.5 * (
        pt_z
        * (
            logvar + z_var[:, None, :] / var - 1 + (z_mu[:, None, :] - mu) ** 2 / var
        ).sum(-1)
    ).sum(-1)

    # E_{q(z,c|x)}[log(p(c)]
    p_c = (pt_z * logpi).sum(1)

    loss = -(px_z + ll_z + p_c + h_c + h_z)

    return loss, px_z + p_c, pt_z


def vae_comp_gmm(x, x_hat, z_mu, z_logvar, mu, logvar, logpi, logvar_x=0.0, eps=1e-8):
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
        :param logvar_x:
        :param eps:

    """

    var = T.exp(logvar)

    # predict the log probability of clusters, shape will be (N, I, C)
    # and compute compute p(t_i|z_i) = p(z_i|t_i)p(t_i)/(\sum_t_i p(z_i|t_i)p(t_i))
    logprob = (
        logpi[:, :, None]
        - 0.5 * (T.log(2 * np.pi) + logvar)
        - (z_mu[:, :, None, :] - mu) ** 2 / (2 * var)
    ).sum(3)
    pt_z = T.softmax(logprob)

    # E_{q(z,c|x)}[log(p(x|z))]
    px_z = ((x - x_hat) ** 2).sum(1)

    # - E_{q(z,c|x)}[log(q(c|x))] entropy of categorical
    h_c = -(pt_z * T.log_softmax(logprob)).sum((1, 2))

    # - E_{q(z,c|x)}[log(q(z|x))] : entropy of normal
    h_z = 0.5 * z_logvar.sum((1, 2))

    # E_{q(z,c|x)}[log(p(z|c)]
    ll_z = -0.5 * (
        pt_z
        * (
            logvar
            + z_var[:, :, None, :] / var
            - 1
            + (z_mu[:, :, None, :] - mu) ** 2 / var
        ).sum(-1)
    ).sum((1, 2))

    # E_{q(z,c|x)}[log(p(c)]
    p_c = (pt_z * logpi[:, :, None]).sum((1, 2))

    loss = -(px_z + ll_z + p_c + h_c + h_z)

    return loss


def FGMM_VAE(
    x,
    x_rec,
    x_logvar,
    z_logvar,
    q_mu,
    q_logvar,
    mu,
    q_loggamma,
    q_logeta,
    logpi,
    logpia,
    mode="bernoulli",
    eps=1e-5,
):
    """N samples of dimension D to latent space of dimension K with F factors of C clusters
    in K dimension

    Parameters
    ----------

    x: array
        (observation) should be of shape (N, D)

    x_rec: array
        (reconstruction, output of decoder) should be of shape (N, D)

    x_logvar: array
        (parameter) should be of shape (D,)

    z_logvar: array
        (parameter) should be of shape (K,)

    q_mu: array
        should be of shape (N, K), infered mean of variational Gaussian

    q_logvar: array
        should be of shape (N, K), infered log-variance of variational Gaussian

    mu: array
        (parameter) should be of shape (F, C, K), parameter (centroids)

    q_loggamma: array
        should be of shape (N, F, C), parameter (prior of clusters)

    logpi: array
        (parameter) should be of shape (F, C), parameter (prior of clusters)

    logpia: array
        (parameters) should ve of shape (F,)
        :param q_logeta:
        :param mode:
        :param eps:

    """

    q_var = T.exp(q_logvar)
    q_gamma = T.exp(q_loggamma)
    q_eta = T.exp(q_logeta)
    z_var = T.exp(z_logvar)
    x_var = T.exp(x_logvar)
    pi = T.exp(logpi)
    pia = T.exp(logpia)

    K = z_var.shape[0]
    D = x_var.shape[0]
    F = logpi.shape[0]
    log2pi = T.log(2 * np.pi)

    # reconstruction part (first expectation)
    E1 = -0.5 * (((x - x_rec) ** 2 / x_var).sum(1) + x_logvar.sum() + D * log2pi)

    E2_1 = -0.5 * (log2pi + z_logvar + (q_var + q_mu ** 2) / z_var).sum(1)

    E2_2 = T.einsum("nf,nfc,fck,nk->n", q_eta, q_gamma, mu, q_mu / z_var)

    E2_3 = -0.5 * (T.einsum("nf,nfc,fck->nk", q_eta, q_gamma, mu ** 2) / z_var).sum(1)

    if mode == "bernoulli":
        q_gammaeta = T.einsum("nf,nfc->nfc", q_eta, q_gamma)
        corr = T.einsum("fcd,nfc,abk,nab->nfa", mu / z_var, q_gammaeta, mu, q_gammaeta)
        E2_4 = -0.5 * T.sum(corr * (1 - T.eye(F)), (1, 2))
    else:
        E2_4 = 0.0
    E3 = (q_gamma * logpi).sum((1, 2))
    if mode == "bernoulli":
        E4 = (q_eta * logpia + (1 - q_eta) * T.log(1 - pia + eps)).sum(1)
    else:
        E4 = (q_eta * logpia).sum(1)

    # now on to the entropy
    H = (
        K * (log2pi + 1) / 2
        + 0.5 * q_logvar.sum(1)
        - (q_gamma * q_loggamma).sum((1, 2))
    )
    if mode == "bernoulli":
        Ha = -(q_eta * q_logeta + (1 - q_eta) * T.log(1 - q_eta + eps)).sum(1)
    else:
        Ha = -(q_eta * q_logeta).sum(1)
    return -(E1 + E2_1 + E2_2 + E2_3 + E2_4 + E3 + E4 + H + Ha)


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
    return -logits * labels + T.log1p(T.exp(logits))


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
        raise TypeError("rank mismatch between targets and predictions")
    corrects = predictions[targets.nonzero()]
    rest = predictions[(1 - targets).nonzero()].reshape((-1, num_cls - 1))
    rest = rest.max(axis=1)
    return T.activations.relu(rest - corrects + delta)


def squared_difference(targets, predictions):
    return (targets - predictions) ** 2


def accuracy(targets, predictions):
    if predictions.ndim == 2:
        accu = T.cast(T.equal(targets, predictions.argmax(1)), "float32")
    else:
        accu = T.cast(T.equal(targets, predictions), "float32")
    return accu.mean()
