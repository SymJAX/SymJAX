import numpy
from .. import tensor

def SGD(params, grads, learning_rate):
    updates = dict()
    for param, grad in zip(params, grads):
        updates[param] = param - learning_rate * grad
    return updates


def Adam(params, grads, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-6):
    updates = dict()

    # Perform Adam
    step = tensor.Variable(0, trainable=False, name='step')
    updates[step] = step + 1

    # get the learning rate
    if not numpy.isscalar(learning_rate) and not isinstance(learning_rate, tensor.Placeholder):
        learning_rate = learning_rate(step)

    def false_fn(m, b, g):
        return m * b + (1 - b)*g

    for param, grad in zip(params, grads):
        m = tensor.Variable(numpy.zeros(param.shape), trainable=False,
                            name='m')
        v = tensor.Variable(numpy.ones(param.shape), trainable=False,
                            name='v')
        updates[m] = tensor.cond(step == 0, grad, lambda x:x, [m, beta1, grad],
                                 false_fn)
        updates[v] = v * beta1 + (1 - beta2)*tensor.square(grad)
        update = learning_rate*updates[m]/(tensor.sqrt(updates[v])+epsilon)
        updates[param] = param - update

    return updates
