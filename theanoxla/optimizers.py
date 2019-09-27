import numpy
from . import tensor

def SGD(params, grads, learning_rate):
    updates = dict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates


def Adam(params, grads, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-6):
    updates = dict()

    # get the learning rate
    if not numpy.isscalar(learning_rate) and not isinstance(learning_rate, tensor.Placeholder):
        learning_rate = learning_rate(step)


    step = tensor.Variable(0., trainable=False, name='step')
    for param, grad in zip(params, grads):
        m, update_m, _ = tensor.ExponentialMovingAverage(grad, beta1, step=step)
        v, update_v, _ = tensor.ExponentialMovingAverage(tensor.square(grad), beta2, step,
                                         init=numpy.zeros(grad.shape))
        updates.update(update_m)
        updates.update(update_v)
        update = updates[m]/(tensor.sqrt(updates[v])+epsilon)
        updates[param] = param - learning_rate * update

    updates[step] = step + 1
    return updates
