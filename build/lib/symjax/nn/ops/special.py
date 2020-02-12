import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from . import Op


class Spectrogram(Op):
    """spectrogram layer (time-frequency representation of a time serie)
    This layer implement a spectrogram computation to allow to include 
    it as part of a model when dealing with time serie data, it is also 
    crucial to allow backpropagation through it in case some previous 
    operation prior to this layer must be differentiated w.r.t. some
    quantities s.a. the loss.

    :param incoming: incoming layer
    :type incoming: Op
    :param window: size of the specgram window
    :type window: int
    :param overlapp: overlapp of the window
    :type overlapp: int
    :param window_func: apodization function
    :type window_func: func
    """
    def __init__(self, incoming, window,overlapp,
                window_func=lambda x:np.ones_like(x)):
        super().__init__(incoming)
        # since we go from a 2d input to 3d input we need to
        # set up by hand the data_format for the next layer
        # we follow same order of channel versus spatial dims
        # we also get the time length

        if incoming.data_format=='NCT':
            self.data_format = 'NCHW'
            self.time_length = self.input_shape[2]
            self.n_channels  = self.input_shape[1]
            self.n_windows   = (time_length-window)//(window-overlapp)+1
            self.output_shape   = (self.input_shape[0],self.n_channels,window,n_windows)
        else:
            self.data_format = 'NHWC'
            self.time_length = self.input_shape[1]
            self.n_windows   = (time_length-window)//(window-overlapp)+1
            self.n_channels  = self.input_shape[2]
            self.output_shape   = (self.input_shape[0],window,n_windows,self.n_channels)
        # If input is a layer
        if self.given_input:
            self.forward(incoming.output,training=training)
    def forward(self,input,training=None,**kwargs):
        if self.data_format=='NCHW':
            input = tf.transpose(input,[0,2,1])

        patches = tf.reshape(tf.extract_image_patches(tf.expand_dims(input,1),
                    [1,1,self.window,1],[1,1,self.hop,1],[1,1,1,1]),
                    [self.input_shape[0],self.n_windows,self.window,self.n_channels])
        output  = tf.abs(tf.rfft(tf.transpose(patches,[0,1,3,2])))
        if self.data_format=='NCHW':
            self.output = tf.transpose(output,[0,2,3,1])
        else:
            self.output = tf.transpose(output,[0,3,1,2])
        self.VQ     = None # to change
        return self.output





class Activation(Op):
    """Apply nonlinearity.

    this layer applies an element-wise nonlinearity to the
    input based on a given scalar to scalar function.
    The nonlinearity can of the following form:

      - a scalar to scalar function :math:`\sigma` leading to
        the output :math:`\sigma(x)`
      - a scalar :math:`\\alpha` , then the activation is defined as
        :math:`\max(x,\\alpha x)`, which thus becomes ReLU :math:`\\alpha=0`,
        leaky-ReLU :math:`\\alpha >0` or absolute value :math:`\\alpha=-1`,
        This corresponds to using a max-affine spline activation function.

    For linear you can use lambda x:x or tf.identity or with :math:`\\alpha=1`.
    We recommand user either one of the last two options for optimize the
    computations.

    Example of use::

        # relu case, recommanded to do it with a scalar for internal
        # optimization, especially if using the backward method later
        # in the graph
        Activation(previous_layer,func_or_scalar = 0)
        # otherwise, it is equivalent to
        Activation(previous_layer,func_or_scalar = tf.nn.relu)

    Parameters
    ----------

    incoming : tf.Tensor or np.ndarray
        the input to the later

    func_or_scalar : scalar or func
        the function to be applied as activation function or the scalar
        that correspond to the slope of the spline for negative inputs.
        For example, -1 for absolute value, 0 for relu and 0.1 for 
        leaky-relu. When using a spline activation as those ones,
        it is recommended to pass a scalar to optimize the backward
        method if used later in the graph.

    """

    _name_ = 'ActivationOp'
    deterministic_behavior = False

    def __init__(self,incoming,func_or_scalar,*args,**kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name           = scope.original_name_scope
            self._func_or_scalar = func_or_scalar
            self.identity        = False
            # determine if a max-affine spline
            if np.isscalar(func_or_scalar):
                if func_or_scalar==1:
                    self.identity = True
                else:
                    self.mas=True
            elif(func_or_scalar==tf.identity):
                self.identity = True
            else:
                self.mas=False
            super().__init__(incoming)

    def forward(self, input,  *args, **kwargs):
        if self.identity: return input
        if np.isscalar(self._func_or_scalar):
            self.mask = tf.greater(input,0)
            output    = tf.maximum(input,self._func_or_scalar*input)
        else:
            output = self._func_or_scalar(input)
        return output
    def backward(self,input,*args,**kwargs):
        if self.identity: return input
        if self.mas:
            return input*tf.cast(self.mask,tf.float32)
        else:
            return tf.gradient(self,self.input,input)[0]




class Inverse(Op):
    """Inverse (backward) operator.
    This operator implements the backward operator given a layer to
    invert and the tensor to feed backward. The inverse term is often
    coined in deep learning even though this simply corresponds to
    standard backpropagation and not an actual inverse operator.
    This operator is usefull when using deconvolutional or other type
    of decoder layers with tied weights w.r.t. theur encoder counterparts.

    Parameters
    ----------

    op_or_layer : sknet.Op or sknet.Layer
        the operator or layer to backprop through

    input : tf.Tensor or sknet.Op or sknet.Layer
        the input to be backpropagated through the derived
        formula obtained form the op_or_layer
    """
    name='Inverse'
    deterministic_behavior=False
    def __init__(self,op_or_layer, input, **kwargs):
        self.op_or_layer = op_or_layer
        super().__init__(input, **kwargs)
    def forward(self,input,**kwargs):
        # if the operator or layer to revert has an implemented
        # backward operator, use it, otherwise
        # use the gradient of it w.r.t. its input.
        # No need to worry about this here as it is already
        # taken care of in the op and layer methods
        return self.op_or_layer.backward(input)






class LambdaFunction(Op):
    """Apply a lambda function onto the input

    This layer allows to apply an arbitrary given function onto
    its input tensor allows to implement arbitrary operations.
    The fiven function must allow backpropagation through it
    if leanring with backpropagation is required, the function
    can alter the shape of the input but a func_shape must be provided
    which outputs the new shape given the tensor one.

    Example of use::

        input_shape = [10,1,32,32]
        # simple case of renormalization of all the values
        # by maximum value
        def my_func(x):
            return x/tf.reduce_max(x)
        layer = LambdaFunction(input_shape, func=my_func, data_format='NCHW')
        # more complex case with shape alteration taking only the first
        # half of the second dimension
        def my_func(x):
            return x[:,:x_shape[1]//2]
        def my_shape_func(x_shape):
            new_shape = x_shape
            new_shape[1]=new_shape[1]//2
            return new_shape

        layer = LambdaFunction(input_shape,func=my_func,
                            shape_func = my_shape_func, data_format='NCHW')

    :param incoming: input shape of tensor
    :type incoming: shape or :class:`Op` instance
    :param func: function to be applied taking as input the tensor
    :type func: func
    :param shape_func: (optional) the function to provide if func 
                       alters the shape of its input. This function
                       takes as input the input shape and outputs the new
                       shape as a list or tuple of ints
    :type shape_func: func
    """
    def __init__(self,incoming,func, shape_func = None, **kwargs):
        super().__init__(incoming, **kwargs)
        self.func = func
        if shape_func is None:
            self.shape_func = lambda x:x
        self.output_shape = self.shape_func(self.input_shape)
        if self.given_input:
            self.forward(input)
    def forward(self,input,**kwargs):
        self.output = self.func(input)
        return self.output
