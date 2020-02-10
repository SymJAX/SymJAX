import tensorflow as tf
import numpy as np
from . import Op



class Dropout(Op):
    """Randomly mask values of the input

    This layer applies a multiplicative perturbation
    on its input by means of a binary mask. Each value
    of the mask is sampled from a Bernoulli distribution
    :math:`\mathcal{B}ernoulli(p)` where :math:`p` is the
    probability to have a :math:`1`.

    Parameters
    ----------

    incoming : :class:`Op` or shape
        the incoming layer or input shape

    p : scalar :math:`0\leq p \leq 1`
        the probability to drop the input values

    """
    _name_ = 'DropoutOp'
    deterministic_behavior = True
    def __init__(self, incoming, p=0.5, deterministic=None,seed=None,
                     **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            self.seed = seed
            assert(np.isscalar(p))
            self.p = p
            super().__init__(incoming, **kwargs)

    def forward(self,input, deterministic, *args, **kwargs):
        r_values = tf.random_uniform(input.shape.as_list(), seed=self.seed)
        mask     = tf.cast(tf.greater(r_values, self.p), input.dtype)
        output   = tf.cond(deterministic, lambda :input, lambda :mask*input)
        return output


class UniformNoise(Op):
    """Applies an additive or multiplicative Uniform noise to the input

    This layer applies an additive or multiplicative perturbation
    on its input by means of a Uniform random variable. Each value
    of the mask is sampled from a Normal distribution
    :math:`\mathcal{U}(lower,upper)` where :math:`lower` and
    :math:`upper` are the bounds of the uniform distirbution
    Those parameters can be constant, per dimension, per channels, ... but
    there shape must be broadcastable to match the input shape.

    Parameters
    ----------

    incoming : :class:`Op` or shape
        the incoming layer or input shape

    noise_type : str, :py:data:`"additive"` or :py:data:`"multiplicative"`
        the type of noise to apply on the input

    lower : Tensor or Array
        the lower bound of the Uniform distribution

    upper : Tensor or Array
        the upper bound of the Uniform distribution

    """

    _name_ = 'UniformNoiseOp'
    deterministic_behavior = True

    def __init__(self, incoming, noise_type="additive", lower = np.float32(0),
            upper = np.float32(1), deterministic=None, seed=None, **kwargs):

        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            # Set attributes
            self.lower = lower
            self.upper = upper
            self.seed  = seed
            self.noise_type = noise_type
            super().__init__(incoming, deterministic=deterministic)
    def forward(self,input, deterministic, *args, **kwargs):
        # Random indices
        r_values = tf.random_uniform(self.input_shape,minval=self.lower,
                                     maxval=self.upper,seed=self.seed)
        if self.noise_type=="additive":
            output = input+r_values
        elif self.noise_type=="multiplicative":
            output = input*r_values
        output = tf.cond(deterministic, lambda :input, lambda :output)
        return output


class GaussianNoise(Op):
    """Applies an additive or multiplicative Gaussian noise to the input

    This layer applies an additive or multiplicative perturbation
    on its input by means of a Gaussian random variable. Each value
    of the mask is sampled from a Normal distribution
    :math:`\mathcal{N}(\mu,\sigma)` where :math:`\mu` is the
    mean and :math:`\sigma` the standard deviation. Those
    parameters can be constant, per dimension, per channels, ... but
    there shape must be broadcastable to match the input shape.

    Parameters
    ----------

    incoming : :class:`Op` or shape
        the incoming layer or input shape

    noise_type : str, :py:data:`"additive"` or :py:data:`"multiplicative"`
        the type of noise to apply on the input

    mu : Tensor or Array
        the mean of the Gaussian distribution

    sigma : Tensor or Array
        the standard deviation of the Gaussian distribution

    """

    _name_ = 'GaussianNoiseOp'
    deterministic_behavior = True

    def __init__(self, incoming, noise_type="additive", mu = np.float32(0),
            sigma = np.float32(1), deterministic=None, seed=None, **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name      = scope.original_name_scope
            self.seed       = seed
            self.mu         = mu
            self.sigma      = sigma
            self.noise_type = noise_type
            super().__init__(incoming, **kwargs)

    def forward(self,input, deterministic, *args, **kwargs):
        # Random indices
        r_values = tf.random_normal(input.get_shape().as_list(), mean=self.mu,
                                         stddev=self.sigma, seed=self.seed)
        if self.noise_type=="additive":
            output = input+r_values
        elif self.noise_type=="multiplicative":
            output = input*r_values
        output = tf.cond(deterministic, lambda :input, lambda :output)
        return output


class RandomContrast(Op):
    """randomly contrast on the input
    """

    _name_ = 'RandomContrastOp'
    deterministic_behavior = True

    def __init__(self, input, vmin, vmax, deterministic=None, seed=None, **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            self.vmin = vmin
            self.vmax = vmax
            self.seed = seed
            super().__init__(input, deterministic=deterministic)

    def forward(self, input, deterministic):
        alpha = tf.random_uniform((input.shape.as_list()[0], 1, 1, 1),
                                  minval=self.vmin, maxval=self.vmax)
        adjusted = alpha*(input-tf.reduce_mean(input, [1, 2, 3],
                                                 keepdims=True))
        adjusted += tf.reduce_mean(input, [1, 2, 3], keepdims=True)
        adjusted = tf.clip_by_value(adjusted, tf.reduce_min(input),
                                    tf.reduce_max(input))
        return tf.cond(deterministic, lambda: input, lambda: adjusted)


class RandomBrightness(Op):
    """randomly contrast on the input
    """

    _name_ = 'RandomBrightnessOp'
    deterministic_behavior = True

    def __init__(self, input, vmin, vmax, deterministic=None, seed=None, **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            self.vmin = vmin
            self.vmax = vmax
            self.seed = seed
            super().__init__(input, deterministic=deterministic)

    def forward(self, input, deterministic):
        alpha = tf.random_uniform((input.shape.as_list()[0], 1, 1, 1),
                                  minval=self.vmin, maxval=self.vmax)
        adjusted = alpha + input
        adjusted = tf.clip_by_value(adjusted, tf.reduce_min(input),
                                    tf.reduce_max(input))
        return tf.cond(deterministic, lambda: input, lambda: adjusted)


class RandomHue(Op):
    """randomly contrast on the input
    """

    _name_ = 'RandomHueOp'
    deterministic_behavior = True

    def __init__(self, input, vmin, vmax, deterministic=None, seed=None, **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            self.vmin = vmin
            self.vmax = vmax
            self.seed = seed
            super().__init__(input, deterministic=deterministic)

    def forward(self, input, deterministic):

        radians = tf.random_uniform((input.shape.as_list()[0], 1),
                                    minval=self.vmin, maxval=self.vmax)
        cosA = tf.cos(radians)
        sinA = tf.sin(radians)
        aa = cosA + (1.0 - cosA) / 3.0
        ab = 1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA
        ac = 1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA
        ba = 1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA
        bb = cosA + 1./3.*(1.0 - cosA)
        bc = 1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA
        ca = 1./3. * (1.0 - cosA) - np.sqrt(1./3.) * sinA
        cb = 1./3. * (1.0 - cosA) + np.sqrt(1./3.) * sinA
        cc = cosA + 1./3. * (1.0 - cosA)

        mat_shape = (input.shape.as_list()[0], 3, 3)
        rot = tf.reshape(tf.stack([aa, ab, ac, ba, bb, bc, ca, cb, cc], 1),
                         mat_shape)
        adjusted = tf.einsum('ncij,nkc->nkij', input, rot)
        adjusted = tf.clip_by_value(adjusted, tf.reduce_min(input),
                                    tf.reduce_max(input))
        return tf.cond(deterministic, lambda: input, lambda: adjusted)


class RandomCrop(Op):
    """Random cropping of input images.

    During deterministic, a random continuous part of the image
    of shape crop_shape is extracted and used as layer output.
    During testing, the center part of the image of shape
    crop_shape is used as output. Apply the same perturbation
    to all the channels of the input.

    Example of use::

        input_shape = (64,3,32,32)
        # Init an input layer with input shape
        crop_layer  = RandomCrop(input_shape,(28,28))
        # output of this layer is (64,3,28,28)
        crop_layer.output_shape


    Parameters
    ----------

    incoming : :class:`Op` or shape
        the incoming layer or input shape

    crop_shape : int or couple of int
        the shape of part of the input to be
        cropped

    """

    _name_ = 'RandomCropOp'
    deterministic_behavior = True

    def __init__(self, input, crop_shape, pad=0, deterministic=None,
                 seed=None):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            # Set attributes
            self.seed = seed
            if np.isscalar(crop_shape):
                self.crop_shape = [crop_shape, crop_shape]
            else:
                assert(len(crop_shape) == 2)
                self.crop_shape = list(crop_shape)
            if np.isscalar(pad):
                self.pad = [pad, pad]
            else:
                assert(len(crop_shape) == 2)
                self.pad = list(pad)

    
            self.spatial_shape = input.shape.as_list()[2:]
            self.spatial_shape[0] += self.pad[0]
            self.spatial_shape[1] += self.pad[1]

            # Number of patches of crop_shape shape in the input in H and W
            self.n_H = np.int64(self.spatial_shape[0]-self.crop_shape[0]+1)
            self.n_W = np.int64(self.spatial_shape[1]-self.crop_shape[1]+1)
    
            super().__init__(input, deterministic=deterministic)

    def forward(self, input, deterministic):

        # Patches form the input:
        # need to transpose to extract patches
        input = tf.pad(input, [[0, 0], [0, 0],
                               [self.pad[0]//2, self.pad[0]//2], 
                               [self.pad[1]//2, self.pad[1]//2]], "REFLECT")
        input_t = tf.transpose(input,[0,2,3,1])
        # Extract patches of the crop_shape shape
        input_patches  = tf.extract_image_patches(input_t,
                                    [1]+self.crop_shape+[1],strides=[1,1,1,1],
                                    rates=[1,1,1,1],padding='VALID')

        # Random indices and patches
        N,C,H,W = input.shape.as_list()
        random_H = tf.random_uniform((N,),maxval=np.float32(self.n_H),
                                                seed=self.seed)
        random_W = tf.random_uniform((N,),maxval=np.float32(self.n_W),
                          seed=self.seed+1 if self.seed is not None else None)
        indices_H = tf.cast(tf.floor(random_H), tf.int32)
        indices_W = tf.cast(tf.floor(random_W), tf.int32)
        random_indices = tf.stack([tf.range(N), indices_H, indices_W],1)
        random_patches = tf.gather_nd(input_patches,random_indices)

        # Deterministic (center) indices and patches
        center_indices = tf.stack([tf.range(N,dtype=tf.int64),
                        tf.fill((N,),self.n_H//2),
                        tf.fill((N,),self.n_W//2)],1)
        center_patches = tf.gather_nd(input_patches,center_indices)

        # Output
        patches = tf.cond(deterministic, lambda :center_patches,
                                    lambda :random_patches)
        # need to reshape as the patches are still flattened
        output  = tf.reshape(patches, [N]+self.crop_shape+[C])
        # transpose back to the original NCHW format
        return tf.transpose(output,[0,3,1,2])


class RandomAxisReverse(Op):
    """randomly reverse axis of the input

    This layer randomly reverse (or flip) one (or multiple) axis
    in its input. It will either apply all the axis or none.
    Apply the same perturbation to all the channels of the input

    Example of use::

        # Set the input shape
        input_shape = [10,1,32,32]
        # Create the layer
        layer = RandomAxisReverse(input_shape,[2],data_format='NCHW')
        # the output will randomly put the images upside down
        # Create another case
        layer = RandomAxisReverse(input_shape,[2,3],data_format='NCHW')
        # in this case the images will randomly have both spatial
        # axis reversed

    Parameters
    ----------
    incoming : Op or shape
        the incoming layer of the input shape

    axis : int or list of int
        the axis to randomly reverse the order on

    deterministic : bool or tf.bool
        the state of the model, can be omited if the layer is not computing
        its output with the constructor
    """

    _name_ = 'RandomAxisReverseOp'
    deterministic_behavior = True

    def __init__(self, input, axis, deterministic=None, seed=None, **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            self.axis  = [axis] if np.isscalar(axis) else axis
            self.seed  = seed
            super().__init__(input, deterministic=deterministic)

    def forward(self, input, deterministic):
        N          = input.shape.as_list()[0]
        prob       = tf.random_uniform((N,), seed=self.seed)
        to_reverse = tf.less(prob,0.5)
        reverse_input = tf.where(to_reverse, tf.reverse(input,self.axis),input)
        output     = tf.cond(deterministic,lambda :input, lambda :reverse_input)
        return output


class RandomRot90(Op):
    """randomly rotate by 90 degrees the input

    This layer performs a random rotation of the input to 90 degrees
    this can be clockwise or counter clockwise with same probability.

    :param incoming: the input shape or the incoming layer
    :type incoming: shape or instalce of :class:`Op`
    :param deterministic: boolean describing if the model is in
                     trianing or testing mode, should be left
                     None in most cases
    :type deterministic: tf.bool
    """
    _name_ = 'RandomRot90Op'
    deterministic_behavior = True
    def __init__(self, incoming, deterministic=None, seed=None, **kwargs):
        with tf.variable_scope(self._name_) as scope:
            self._name = scope.original_name_scope
            self.output_shape = self.input_shape
            self.seed = seed
            super().__init__(incoming, deterministic=deterministic)
    def forward(self,input, deterministic, **kwargs):
        N = input.shape.as_list()[0]
        r_values = tf.random_uniform((N,), maxval=np.float32(3), seed=self.seed)
        rot_left  = tf.less(r_values,1)
        rot_right = tf.greater(r_values,2)
        left_rot    = tf.transpose(input,[0,1,3,2])
        right_rot   = tf.reverse(left_rot,[-1])

        left_images = tf.where(self.rot_left,left_rot,input)
        output      = tf.where(self.rot_right,right_rot, left_images)
        output      = tf.cond(deterministic, lambda :input, lambda :output)
        return output

