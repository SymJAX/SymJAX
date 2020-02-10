import numpy

from ... import tensor
from ..base import Layer


class Dense(Layer):
    """Dense or fully connected layer.
    This layer implement a fully connected (or dense) linear operator.

    Parameters
    ----------

    incoming : tf.Tensor or numpy.ndarray
        inumpy.t to the dense operator, can be another 
        :py:class:~`sknet.layer.Op`, a :py:class:`tf.placeholder`, or even
        a :py:class:`numpy.ndarray`.

    units : int
        the number of output units, must be greater or equal to 1.

    W : func or tf.Tensor or numpy.ndarray
        the initialization for the :math:`W` parameter. In any case, if the
        given value is not a function then it is considered a not trainable
        (because coming from another computation and thus not to be used
        solely as initialization of a trainable unconstrained variable). If
        a func then it is assumed to be trainable. The function will be passed
        the shape of the parameter and should return the value that will be
        used for initialization of the variable.
        Example of use ::
            
            # the case where the W parameter will be treated as a given
            # external (and thus non trainable in the layer) parameter
            # assuming previous_layer is 2D
            W = tf.random_normal((previous_layer.shape[1],20))
            Dense(previous_layer,units=20,W = W)
            # to have a learnable (unconstrained) parameter simply do
            # if already given the value
            Dense(previous_layer,units=20,W = lambda *x:W)
            # or in this case, simply do
            Dense(previous_layer,units=20,W = tf.random_normal)

    b : func or tf.Tensor or numpy.ndarray
        same than W but for the :math:`b` parameter.

    func_W : (optional, default = tf.identity) func 
        an external function to be applied onto the weights :math:`W` before
        computing the forward pass
        Example of use ::
            
            # this will force the weights W to be nonnegative to
            # compute the layer output (all nonegative weights
            # are turned to 0 by the relu applied on W)
            Dense(previous_layer,units=20,func_W=tf.nn.relu)

    func_b : (optional, default = tf.identity) func
        same as func_W but for the :math:`b` parameter

    """
    _name_ = 'DenseOp'
    deterministic_behavior = False

    def __init__(self, incoming, units, W=numpy.ones , b=numpy.zeros,
                 W_func = tensor.identity, b_func = tensor.identity, name=''):
        # Set up the inumpy.t, flatten if needed
        if incoming.ndim > 2:
            self._flatten_input = True
            flat_dim  = numpy.prod(incoming.shape[1:])
        else:
            self._flatten_input = False
            flat_dim  = incoming.shape[1]

        # Initialize W
        if callable(W):
            self._W = tensor.Variable(W((flat_dim, units)), name='W')
        else:
            self._W = W
        self.W = W_func(self._W)

        # Initialize b
        if b is None:
            self.b  = None
        else:
            if callable(b):
                self._b = tensor.Variable(b((1,units)), name=name+'b')
            else:
                self._b = b
            self.b  = b_func(self._b)
        super().__init__(dense, [incoming, self.W, self.b], [self.W, self.b])



def dense(input, W, b):
        flattened  = tensor.flatten2d(input) if input.ndim > 2 else input
        return tensor.matmul(flattened, W) + b




