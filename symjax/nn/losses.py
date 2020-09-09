from .. import tensor as T
import numpy as np
from .. import probabilities
from .ops_nn import softmax, log_softmax


def huber(targets, predictions, delta=1.0):
    """huber loss (regression).

    For each value x in `error=targets-predictions`, the following is calculated:

        - :math:`0.5 × x^2`     if :math:`|x| <= Δ`
        - :math:`0.5 × Δ^2 + Δ × (|x| - Δ)`  if :math:`|x| > Δ`

    leading to

    .. plot::

      import matplotlib.pyplot as plt
      import numpy as np
      x = np.linspace(-3, 3, 300)
      huber = lambda x, delta: np.where(np.abs(x)<= delta, 0.5*x**2, 0.5*delta**2+delta*(np.abs(x)-delta))
      plt.plot(x, huber(x, 0.5))
      plt.plot(x, huber(x, 1))
      plt.plot(x, huber(x, 2))
      plt.plot(x, x**2, '--k')
      plt.xlabel(r'$x$')
      plt.ylabel(r'huber$(x,\Delta)$')
      plt.legend([r'$\Delta=0.5$',r'$\Delta=1$',r'$\Delta=2$', r'$x^2$'])
      plt.tight_layout()
      plt.show()


    `Wikipedia <https://en.wikipedia.org/wiki/Huber_loss>`_


    Args:
      targets: The ground truth output tensor, same dimensions as 'predictions'.
      predictions: The predicted outputs.
      delta (Δ): `float`, the point where the huber loss function changes from a
        quadratic to linear.
    Returns:
      loss float, this has the same
      shape as `targets`
    """

    error = predictions - targets
    abs_error = T.abs(error)
    losses = T.where(
        abs_error <= delta, 0.5 * error ** 2, delta * (abs_error - 0.5 * delta)
    )
    return losses


def explained_variance(y, ypred, axis=None, epsilon=1e-6):
    """
    Computes fraction of variance that ypred explains about y.
    The formula is

    .. math::
        1 - Var[y-ypred] / Var[y]

    and in the special case of centered targets and predictions it becomes

    .. math::
        1 - \|y-ypred\|^2_2 / \|y\|_2^2

    hence it can be seen as an :math:`ℓ_2' loss rescaled by the energy in the targets.

    interpretation:

        - ev=0  =>  might as well have predicted zero
        - ev=1  =>  perfect prediction
        - ev<0  =>  worse than just predicting zero

    Parameters:
    -----------

    y: Tensor like
        true target

    ypred: Tensor like
        prediction

    axis: integer or None (default=None)
        the axis along which to compute the var, by default uses all axes

    epsilon (ϵ): float (default=1e-6)
        the added constant in the denominator

    Returns:
    --------

    .. math::
        1 - Var(y-ypred)/(Var(y)+ϵ)

    Notes:
    ------

    This is not a symmetric function

    """
    return 1 - (y - ypred).var(axis=axis) / (y.var(axis=axis) + epsilon)


def vae(x, x_hat, z_mu, z_logvar, mu, logvar, x_logvar=0.0):
    """N samples of dimension D to latent space in K dimension with Gaussian distributions

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
        should be of shape (K,), mean of z variable

    logvar: array
        should be of shape (K,), logstd of z variable

    """

    p = probabilities.MultivariateNormal(x_hat, logstd=x_logvar)
    q = probabilities.MultivariateNormal(z_mu, logstd=z_logvar)
    z = probabilities.MultivariateNormal(mu, logstd=logvar)

    loss = -(p.logprob(x) + probabilities.KL(z, q) + q.entropy())

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
    z_var = T.exp(z_logvar)

    # predict the log probability of clusters, shape will be (N, I, C)
    # and compute compute p(t_i|z_i) = p(z_i|t_i)p(t_i)/(\sum_t_i p(z_i|t_i)p(t_i))
    logprob = (
        logpi[:, :, None]
        - 0.5 * (T.log(2 * np.pi) + logvar)
        - (z_mu[:, :, None, :] - mu) ** 2 / (2 * var)
    ).sum(3)
    pt_z = softmax(logprob)

    # E_{q(z,c|x)}[log(p(x|z))]
    px_z = ((x - x_hat) ** 2).sum(1)

    # - E_{q(z,c|x)}[log(q(c|x))] entropy of categorical
    h_c = -(pt_z * log_softmax(logprob)).sum((1, 2))

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
    p_c = (pt_z * logpi).sum((1, 2))

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


def sparse_softmax_crossentropy_logits(p, q):
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
    # one = T.equal(p[:, None], T.arange(q.shape[1])).astype("float32")
    # return -(one * log_softmax(q)).sum(1)
    return -T.take_along_axis(log_softmax(q), p[:, None], 1).squeeze(1)


def softmax_crossentropy_logits(p, q):
    """see sparse cross entropy"""
    return -(p * log_softmax(q)).sum(-1)


def sigmoid_crossentropy_logits(labels, logits):
    return -logits * labels + T.log1p(T.exp(logits))


def hinge_loss(predictions, targets, delta=1):
    """(binary) hinge loss.


    For an intended output :math:`t = ±1` and a classifier score :math:`p`, the hinge loss is defined for each datum as

    .. math::
        \max ( 0 , Δ − t  p)

    as soon as the loss is smaller than :math:`Δ` the datum is well classified, however margin is increased by pushing the loss to :math:`0` hence :math:`Δ` is the user-defined prefered margin to reach. In standard SVM :math:`Δ=1`
    leading to

    .. plot::

      import matplotlib.pyplot as plt
      import numpy as np
      x = np.linspace(-3, 3, 300)
      y = 1
      hinge1 = np.maximum(0, 1-x*y)
      hinge02 = np.maximum(0, 0.2-x*y)
      plt.plot(x, hinge1)
      plt.plot(x, hinge02)
      plt.plot(x, (x < 0).astype('int32'), '--k')
      plt.axvline(1,c='r')
      plt.xlabel(r'predictions')
      plt.ylabel(r'hinge$(p,1,\Delta)$')
      plt.legend([r'hinge loss $\Delta=1$',r'hinge loss $\Delta=0.2$' '0-1 loss', 'true label'])
      plt.tight_layout()
      plt.show()


    Note that :math:`p` should be the "raw" output of the classifier's decision function, not the predicted class label. For instance, in linear SVMs, :math:`p = <w, x> + b` where ( :math:`w , b` are the parameters of the hyperplane and :math:`x` is the input variable(s).

    Parameters
    ----------
    predictions : 1D tensor
        prediction of the classifier (raw,)
    targets : 1D binary tensor with values in :math:`t\in\{-1,1\}`.

    Returns
    -------
    1D tensor
        An expression for the item-wise hinge loss

    Notes
    -----
    This is an alternative to the categorical cross-entropy loss for
    classification problems
    """
    return T.maximum(0, delta - predictions * targets)


def multiclass_hinge_loss(predictions, targets, delta=1):
    """multi-class hinge loss.

    .. math::
        L_i = \max_{j ≠ t_i} (0, p_j - p_{t_i} + Δ)
    Parameters
    ----------
    predictions : 2D tensor
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


def squared_differences(x, y):
    """elementwise squared differences.

    Computes element-wise

    .. math::

        (x-y)^2

    broadcasting applies as in any
    operations.

    `Wikipedia <https://en.wikipedia.org/wiki/Loss_function#Quadratic_loss_function>`_

    Parameters
    ----------

    x: tensor-like

    y: tensor-like

    Returns
    -------

    tensor-like

    """

    return (x - y) ** 2


def accuracy(targets, predictions):
    """classification accuracy.

    It is computed by averaging the `0-1 loss <https://en.wikipedia.org/wiki/Loss_function#0-1_loss_function>`_
    as in

    .. math::
        (Σ_{n=1}^N 1_{\{y_n == p_n\}})/N

    where :math:`p` denotes the predictions. The inputs must be vectors but in
    the special case where targets is a vector
    but predictions is a matrix, then the
    argmax is used to get the real predictions as in

    .. math::
        (Σ_{n=1}^N 1_{\{y_n == arg \max p_{n,:}\}})/N

    `Wikipedia <https://en.wikipedia.org/wiki/Accuracy_and_precision>`_

    Parameters
    ----------

    targets: 1D tensor-like

    predictions: tensor-like
        it can be a :math:`2D` matrix in which case the ``argmax`` is used to
        get the prediction

    Returns
    -------

    tensor-like
    """

    if not hasattr(targets, "ndim"):
        targets = T.array(targets)

    if targets.ndim != 1:
        raise RuntimeError("targets should be of rank 1, given rank is {targets.ndim}")

    if predictions.ndim == 2:
        accu = T.cast(T.equal(targets, predictions.argmax(1)), "float32")
    elif predictions.ndim == 1:
        accu = T.cast(T.equal(targets, predictions), "float32")
    else:
        raise RuntimeError(
            "predictions should be of rank 1 or 2, given rank is {predictions.ndim}"
        )
    return accu.mean()


def _assign(cluster, pred, true):
    c_labels = T.where(T.equal(pred, cluster)[:, None], true, 0)
    per_cluster_counts = c_labels.sum(0)
    assignment = per_cluster_counts.argmax()
    return assignment


def clustering_accuracy(labels, predictions, n_clusters):
    """
    find accuracy of clustering based on intra cluster labels

    This accuracy allows to quantify the ability of a clustering algorithm to solve the clustering task
    given the true labels of the data.
    This functions finds for each predicted cluster
    what is the most present label and uses it as the
    cluster label. Based on those cluster labels the
    accuracy is then computed.

    Args:

    labels: 1d integer Tensor
        the true labels of the data

    predictions: 1d integer Tensor
        the predicted data clusters

    n_clusters: int
        the number of clusters

    """
    one_hot_labels = T.one_hot(labels, n_clusters)
    cluster_assignment = T.map(
        _assign,
        sequences=[T.range(n_clusters, dtype="int32")],
        non_sequences=[predictions, one_hot_labels],
    )
    return accuracy(labels, cluster_assignment[predictions])
