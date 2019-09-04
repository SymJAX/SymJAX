import tensorflow as tf
import tensorflow.contrib.layers as tfl
from .normalize import BatchNorm as bn
from .special import Activation as sa
#from .special import Identity
import numpy as np

from . import Op



class Dense(Op):
    """Dense or fully connected layer.
    This layer implement a fully connected (or dense) linear operator.

    Parameters
    ----------

    incoming : tf.Tensor or np.ndarray
        input to the dense operator, can be another 
        :py:class:~`sknet.layer.Op`, a :py:class:`tf.placeholder`, or even
        a :py:class:`np.ndarray`.

    units : int
        the number of output units, must be greater or equal to 1.

    W : func or tf.Tensor or np.ndarray
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

    b : func or tf.Tensor or np.ndarray
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

    def __init__(self, incoming, units, W = tfl.xavier_initializer(), 
                b = tf.zeros, W_func = tf.identity, 
                b_func = tf.identity, name='',*args, **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            # Set up the input, flatten if needed
            if len(incoming.shape.as_list())>2:
                self._flatten_input = True
                flat_dim  = np.prod(incoming.shape.as_list()[1:])
            else:
                self._flatten_input = False
                flat_dim  = incoming.shape.as_list()[1]

            # Initialize W
            if callable(W):
                self._W = tf.Variable(W((flat_dim,units)),name='W')
            else:
                self._W = W
            self.W = W_func(self._W)
            # Initialize b
            if b is None:
                self.b  = None
            else:
                if callable(b):
                    self._b = tf.Variable(b((1,units)),name=name+'b')
                else:
                    self._b = b
                self.b  = b_func(self._b)

            super().__init__(incoming)

    def forward(self, input, *args, **kwargs):
        if self._flatten_input:
            input = tf.layers.flatten(input)
        if self.b is None:
            return tf.matmul(input,self.W)
        else:
            return tf.matmul(input,self.W)+self.b
    def backward(self,input,*args, **kwargs):
        output = tf.matmul(input,self.W,transpose_b=True)
        if self._flatten_input:
            return tf.reshape(output,self._input.shape)
        else:
            return output






class Dense2:
    """Dense or fully connected layer.
    This layer implement a fully connected or a dense layer with or without
    nonlinearity, and batch-norm

    :param incoming: input shape or incoming :class:`Op` instance
    :type incoming: tuple of int or Op
    :param units: then umber of output units (or neurons)
    :type units: int
    :param nonlinearity_c: the coefficient for the nonlinearity, 
                           0 for ReLU, -1 for absolute value, ...
    :type nonlinearity_c: scalar
    :param deterministic: a dummy Tensorflow boolean stating if it is 
                     deterministic time or testing time
    :type deterministic: tf.bool
    :param batch_norm: using or not the batch-normalization
    :type batch_norm: bool
    :param W: initialization for the W weights
    :type W: initializer of tf.tensor or np.array
    :param b: initialization for the b weights
    :type b: initializer of tf.tensor or np.array
    :param name: name for the layer
    :type name: str

    """
    variables=["W","b","output"]
    def __init__(self, incoming, units, nonlinearity = np.float32(1),
                deterministic=None, batch_norm = False,
                W = tfl.xavier_initializer(uniform=True), 
                b = tf.zeros, observed=[],observation=[],
                teacher_forcing=[],name=''):
        if len(observed)>0:
            for obs in observed:
                assert(obs in Dense.variables)
        # Set up the input, flatten if needed
        if len(incoming.shape.as_list())>2:
            self._flatten_input = True
            flat_dim  = np.prod(incoming.shape.as_list()[1:])
        else:
            self._flatten_input = False
            flat_dim  = incoming.shape.as_list()[1]

        # Initialize the layer variables
        self._W = init_var(W,(flat_dim,units),
                            name='dense_W_'+name,
                            trainable=True)
        self._b = init_var(b,(1,units),
                            trainable=True,
                            name='dense_b_'+name)
        if "W" in observed:
            self.W = Tensor(self._W,observed=True,)

        super().__init__(incoming, deterministic=deterministic, 
                        observed=observed, observation=observation, 
                        teacher_forcing=teacher_forcing)

    def forward(self, input, deterministic=None, **kwargs):
        if self._flatten_input:
            input = tf.layers.flatten(input)
        return tf.matmul(input,self._W)+self._b




class ConstraintDenseOp:
    def __init__(self,incoming,n_output,constraint='none',deterministic=None):
        # bias_option : {unconstrained,constrained,zero}
        if(len(incoming.output_shape)>2): reshape_input = tf.layers.flatten(incoming.output)
        else:                             reshape_input = incoming.output
        in_dim      = prod(incoming.output_shape[1:])
        self.gamma  = tf.Variable(ones(1,float32),trainable=False)
        gamma_update= tf.assign(self.gamma,tf.clip_by_value(tf.cond(deterministic,lambda :self.gamma*1.005,lambda :self.gamma),0,60000))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,gamma_update)
        W      = tf.contrib.layers.xavier_initializer(uniform=True)
        if(constraint=='none'):
                self.W_     = tf.Variable(W((in_dim,n_output)),name='W_dense',trainable=True)
                self.W      = self.W_
        elif(constraint=='dt'):
                self.W_     = tf.Variable(W((in_dim,n_output)),name='W_dense',trainable=True)
                self.alpha  = tf.Variable(randn(1,n_output).astype('float32'),trainable=True)
                self.W      = self.alpha*tf.nn.softmax(tf.clip_by_value(self.gamma*self.W_,-20000,20000),axis=0)
        elif(constraint=='diag'):
                self.sign   = tf.Variable(randn(in_dim,n_output).astype('float32'),trainable=True)
                self.alpha  = tf.Variable((randn(1,n_output)/sqrt(n_output)).astype('float32'),trainable=True)
                self.W      = tf.nn.tanh(self.gamma*self.sign)*self.alpha
        self.output_shape = (incoming.output_shape[0],n_output)
        self.output       = tf.matmul(reshape_input,self.W)
        self.VQ = None










