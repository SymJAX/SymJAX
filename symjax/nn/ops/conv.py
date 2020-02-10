import tensorflow as tf
import tensorflow.contrib.layers as tfl
import numpy as np
from . import Op

from .. import utils


class Conv1D(Op):
    """1D (temporal) convolutional layer.
    Op to perform a 1D convolution onto a 3D input tensor

    :param incoming: input shape of incoming layer
    :type incoming: Op or tuple of int
    :param filters: the shape of the filters in the form
                    (#filters, width)
    :type filters: triple of int
    :param nonlinearity_c: coefficient of the nonlinearity,
                           0 for ReLU,-1 for absolute value,...
    :type nonlinearity_c: scalar

    """
    _name_ = 'Conv1DOp'
    deterministic_behavior = False
    def __init__(self,incoming,filters,W = tfl.xavier_initializer(),
                    b = tf.zeros, stride =1, pad='valid',
                    mode='CONSTANT', name='', W_func = tf.identity,
                    b_func = tf.identity, separable=False, *args, **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name   = scope.original_name_scope
            self.mode    = mode
            self.stride  = stride
            self.pad     = pad
            self.stride = stride
            self.separable = separable
            # Define the padding function
            if pad=='valid' or (filters[1]==1):
                self.to_pad=False
            else:
                if pad=='same':
                    assert(filters[1]%2==1 and filters[2]%2==1)
                    self.padding = [[0,0], [0,0], [(filters[1]-1)//2]*2]
                else:
                    self.padding = [[0,0], [0,0], [filters[1]-1]*2]
                self.to_pad = True

            # Compute shape of the W filter parameter
            if separable:
                w_shape = (filters[1], 1, filters[0])
            else:
                w_shape = (filters[1], incoming.shape.as_list()[1], filters[0])

            # Initialize W
            self._W = tf.Variable(W(w_shape), name='W') if callable(W) else W
            self.W  = W_func(self._W)

            # Initialize b
            if b is None:
                self._b  = None
            elif callable(b):
                self._b = tf.Variable(b((filters[0],1)), name='b')
            else:
                self._b = b
            self.b  = b_func(self._b) if b is not None else self._b

            super().__init__(incoming)

    def forward(self,input, *args,**kwargs):
        # Reshape in case the input already has multi-channels
        input_shape = input.shape.as_list()
        if self.separable:
            new_shape = [np.prod(input_shape[:-1]),1,input_shape[-1]]
            output_shape = input_shape[:-1]+[None,None]
        else:
            new_shape = [np.prod(input_shape[:-2])]+input_shape[-2:]
            output_shape = input_shape[:-2]+[None,None]

        if new_shape!=input_shape:
            input = tf.reshape(input,new_shape)

        if self.to_pad:
            input = tf.pad(input, self.padding, mode=self.mode)
        output = tf.nn.conv1d(input, self.W, stride=self.stride,
                                      padding='VALID', data_format="NCW")
        conv_shape = output.shape.as_list()
        output_shape[-2:]=conv_shape[-2:]
        if output_shape!=conv_shape:
            output = tf.reshape(output,output_shape)
        return output if self.b is None else output+self.b

    def backward(self, input):
        return tf.gradients(self, self.input, input)[0]


class Conv2D(Op):
    """2D (spatial) convolutional layer.
    Op to perform a 2D convolution onto a 4D input tensor

    :param incoming: input shape of incoming layer
    :type incoming: Op or tuple of int
    :param filters: the shape of the filters in the form 
                    (#filters, height, width)
    :type filters: triple of int
    :param nonlinearity_c: coefficient of the nonlinearity,
                           0 for ReLU,-1 for absolute value,...
    :type nonlinearity_c: scalar

    """
    _name_ = 'Conv2DOp'
    deterministic_behavior = False
    def __init__(self, incoming, filters, W=tfl.xavier_initializer(),
                 b=tf.zeros, strides=1, pad='valid',
                 mode='CONSTANT', name='', W_func=tf.identity,
                 b_func=tf.identity):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            self.mode = mode
            self.strides = strides
            self.pad = pad
            if np.isscalar(strides):
                self.strides = [1, 1, strides, strides]
            else:
                self.strides = [1, 1]+list(strides)

            # Define the padding function
            if pad == 'valid' or (filters[1] == 1 and filters[2] == 1):
                self.to_pad = False
            else:
                if pad == 'same':
                    assert(filters[1]%2==1 and filters[2]%2==1)
                    self.p = [[0]*2, [0]*2, [(filters[1]-1)//2]*2,
                                                  [(filters[2]-1)//2]*2]
                else:
                    self.p = [[0]*2, [0]*2, [filters[1]-1]*2, [filters[2]-1]*2]
                self.to_pad = True

            # Compute shape of the W filter parameter
            w_shape = (filters[1], filters[2],
                       incoming.shape.as_list()[1], filters[0])
            # Initialize W
            if callable(W):
                self._W = tf.Variable(W(w_shape), trainable=True, name='W')
            else:
                self._W = W
            self.W = W_func(self._W)

            # Initialize b
            if b is None:
                self._b = None
            elif callable(b):
                self._b = tf.Variable(b((filters[0], 1, 1)), trainable=True,
                                      name='b')
            else:
                self._b = b
            self.b = b_func(self._b) if b is not None else self._b

            super().__init__(incoming)

    def forward(self, input, *args, **kwargs):
        padded = tf.pad(input, self.p, mode=self.mode) if self.to_pad else input
        Wx = tf.nn.conv2d(padded, self.W, strides=self.strides,
                          padding='VALID', data_format="NCHW")
        return Wx if self.b is None else Wx+self.b

    def backward(self, input):
        return tf.nn.conv2d_backprop_input(self.input.get_shape().as_list(),
                                           filter=self.W, out_backprop=input,
                                           strides=self.strides,
                                           data_format='NCHW', padding='VALID')




#######################################
#
#       1D spline
#




class HermiteSplineConv1D(Op):
    """Learnable scattering network layer.

    Parameters
    ----------

    J : int
        The number of octave (from Nyquist) to decompose.

    Q : int
        The resolution (number of wavelets per octave).

    K : int
        The number of knots to use for the spline approximation of the
        mother wavelet. Should be odd, if not, will be rounded to the lesser
        odd value.

    strides : int (default 1)
        The stride for the 1D convolution.

    init : str (default gabor)
        The initialization  for the spline wavelet, can be :data:`"gabor"`,
        :data:`"random"`, :data:`"paul"`.

    trainable_scales : bool (default True)
        If the scales (dilation of the mother wavelet) should be learned

    trainable_knots : bool (default True)
        If the knots (position for the spline region boundaries) should be
        learned
    """
    deterministic_behavior = False
    _name_ = 'SplineWaveletTransformOp'

    def __init__(self, input, J, Q, K, strides=1, init='random',
                 trainable_scales=0, trainable_knots=False, padding='SAME',
                 trainable_filters=False, hilbert=False, m=None,
                 p=None, n_conv=None, tied_knots=True, trainable_chirps=False,
                 complex=True, tied_weights=True, **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            n_conv = 1 if n_conv is None else n_conv
            K += (K%2)-1
            self.padding           = padding
            self.J,self.Q,self.K   = J, Q, K
            self.trainable_chirps  = trainable_chirps
            self.trainable_scales  = trainable_scales
            self.trainable_knots   = trainable_knots
            self.trainable_filters = trainable_filters
            self.hilbert           = hilbert
            self.strides           = strides
            self.complex           = complex
            self.tied_weights      = tied_weights
            self.tied_knots        = tied_knots
            self.init = init
            # ------ SCALES
            # we start with scale 1 for the nyquist and then increase 
            # the scale for the lower frequencies. This is built by using
            # a standard dyadic scale and then adding a (learnable) vector 
            # to it. As such, regularizing this delta vector constrains the 
            # learned scales to not be away from standard scales, we then 
            # sort them to have an interpretable time/frequency plot and to 
            # have coherency in case this is followed by 2D conv.
            if trainable_scales<2:
                scales = 2**(tf.range(self.J,delta=1./self.Q,dtype=tf.float32))
                scales = tf.Variable(scales, trainable=self.trainable_scales,
                                                            name='scales')
                sorted_scales = tf.contrib.framework.sort(scales)
            else:
                Q_ = tf.Variable(np.float32(self.Q), trainable=True, name='Q')
                b = tf.Variable(np.float32(0), trainable=True, name='b')
                scales = 2**(b+tf.range(self.J*self.Q,dtype=tf.float32)/Q_)
                sorted_scales = scales

            self.scales  = sorted_scales
            self.indices = np.arange(0,J,1./self.Q)

            # We initialize the knots  with uniform spacing 
            grid  = tf.range(self.K,dtype=tf.float32)-(self.K//2)
            if tied_knots:
                knots = tf.Variable(grid, trainable=self.trainable_knots,
                                                                name='knots')
                self.knots = tf.contrib.framework.sort(knots)
                self.all_knots = tf.einsum('i,j->ij',self.scales,self.knots)
            else:
                grid_2d = tf.expand_dims(grid,0)*tf.ones((J*Q,1))
                knots = tf.Variable(grid_2d, trainable= self.trainable_knots,
                                                               name='knots')
                self.knots = tf.contrib.framework.sort(knots)
                self.all_knots =tf.expand_dims(self.scales,1)*self.knots

            # initialize m and p
            self.init_mp(m,p)

            # create the n_conv filter-bank(s)
            self.W  = [self.init_filters(i*J*Q//n_conv,(i+1)*J*Q//n_conv)
                                    for i in range(n_conv)]

            super().__init__(input)

    def forward(self,input,*args,**kwargs):
        # Reshape in case the input already has multi-channels
        input_shape = input.shape.as_list()
        new_shape = [np.prod(input_shape[:-1]),1,input_shape[-1]]
        if new_shape!=input_shape:
            input = tf.reshape(input,new_shape)

        # define the output shape for the first dims.
        if len(input_shape)==2 or (len(input_shape)==3 and input_shape[1]==1):
            output_shape=input_shape[:1]+[None,None]
        else:
            output_shape=input_shape[:-1]+[None,None]

        outputs = list()
        for i in range(len(self.W)):
            # shape of W is (width,inchannel,outchannel)
            if self.padding=='SAME':
                width = self.W[i].shape.as_list()[0]
                output_shape[-2:]=[self.J*self.Q,input_shape[-1]//self.strides]

                amount_l = np.int32(np.floor((width-1)/2))
                amount_r = np.int32(width-1-amount_l)
                paddings = [[0,0],[0,0],[amount_l,amount_r]]
                x_pad = tf.pad(input, paddings=paddings, mode='SYMMETRIC')

                conv = self.apply_filter_bank(x_pad,self.W[i])
                outputs.append(tf.reshape(conv,output_shape))
            else:
                width0 = self.W[-1].shape.as_list()[0]
                time_bins = (input_shape[-1]-width0)//self.strides+1
                output_shape[-2:] = [out_c,time_bins]

                width, in_c, out_c = self.W[i].shape.as_list()[0]
                width_diff = width0-width
                amount_l = np.int32(np.floor(width_diff/2))

                conv = self.apply_filter_bank(input,self.W[i])
                conv = conv[...,amount_l:amount_l+output_shape[-1]]
                outputs.append(tf.reshape(conv,output_shape))

        if len(self.W)>1:
            output = tf.concat(outputs,-2)
        else:
            output = outputs[0]
        return output


    def apply_filter_bank(self,input,W):
        conv = tf.nn.conv1d(input,W,stride=self.strides,
                                        padding='VALID',data_format='NCW')
        if self.complex:
            return tf.complex(conv[...,:self.J*self.Q//len(self.W),:],
                                   conv[...,self.J*self.Q//len(self.W):,:])
        return conv

    def init_mp(self, m, p):
        if m is not None and p is not None:
            self._m = m
            self._p = p
        else:
            # the filters can be (1,K), (2,K), (J*Q,K), (2*J*K,K)
            B = 1 if self.tied_weights else self.J*self.Q
            if self.init=='gabor':
                window = np.hamming(self.K)
                m = (np.cos(np.arange(self.K)*np.pi)*window).astype('float32')
                m = np.ones((1,B,1),dtype='float32')*m
                p = np.zeros((1,B,self.K),dtype='float32')
                if self.complex and not self.hilbert:
                    m_imag = np.zeros((1, B, self.K))
                    p_imag = np.cos(np.arange(self.K) * np.pi)*window
                    p_imag = p_imag*np.ones((1, B, 1))
                    m = np.concatenate([m,m_imag],0).astype('float32')
                    p = np.concatenate([p,p_imag],0).astype('float32')
            elif self.init=='random':
                m = np.random.randn(1, B, self.K).astype('float32')
                p = np.random.randn(1, B, self.K).astype('float32')
                if self.complex and not self.hilbert:
                    m_imag = np.random.randn(1, B, self.K)
                    p_imag = np.random.randn(1, B, self.K)
                    m = np.concatenate([m, m_imag],0).astype('float32')
                    p = np.concatenate([p, p_imag],0).astype('float32')
            self._m = tf.Variable(m, trainable=self.trainable_filters,name='m')
            self._p = tf.Variable(p, trainable=self.trainable_filters,name='p')
        self.m_channels = self._m.shape.as_list()[1]
        # Boundary Conditions and centering
        mask    = np.ones((1,1,self.K), dtype=np.float32)
        mask[0,0,0], mask[0,0,-1] = 0, 0
        m_null = self._m - tf.reduce_mean(self._m[...,1:-1], axis=-1,
                                                    keepdims=True)
        self.m = m_null*mask
        self.p = self._p*mask


    def init_filters(self,start=0,end=-1):
        """
        Method initializing the filter-bank
        """

        # ------ TIME SAMPLING
        # add an extra octave if learnable scales (to go to lower frequency)
        # Define the integer time grid (time sampling) 
        length    = int(self.K*2**(self.indices[end-1]
                                        +int(self.trainable_scales)))
        time_grid = tf.linspace(np.float32(-(length//2)),
                                    np.float32(length//2), length)

        # ------ FILTER-BANK
        if self.hilbert:
            exit()
            filters_real = utils.hermite_interp(time_grid,
                        self.all_knots[start:end], self.m, self.p, True)
            filters_fft = tf.spectral.rfft(filters)
            filters = tf.ifft(tf.concat([filters_fft,
                                            tf.zeros_like(filters_fft)],1))
        else:
            if self.m_channels==1:
                filters = utils.hermite_interp(time_grid,
                                    self.all_knots[start:end], self.m, self.p)
            else:
                filters = utils.hermite_interp(time_grid,
                                     self.all_knots[start:end],
                                     self.m[:,start:end], self.p[:,start:end])
            if self.complex:
                new_shape = (2*len(self.indices[start:end]),1,length)
                filters = tf.reshape(filters,new_shape)
                W = tf.transpose(filters,[2,1,0])
                W.set_shape(new_shape[[2,1,0]])
            else:
                W = tf.transpose(filters,[2,0,1])
                W.set_shape((length,1,len(self.indices[start:end])))
        return W


