import tensorflow as tf


def count_number_of_params():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


def min_distance(S,A):
    return tf.abs(S)/tf.sqrt(tf.reduce_sum(tf.square(A),1))


def get_distance(layers):
    identity = tf.eye(layers[-1].output_shape[1])
    input    = layers[0].output
    S        = layers[-1].S
    dist     = tf.map_fn(lambda v:min_distance(tf.reduce_sum(S*tf.expand_dims(v,0),1),tf.gradients(S*tf.expand_dims(v,0),input)[0]),identity) # (output,batch)
    return tf.reduce_min(dist,0)







def batch_normalization(tensor,axis,training,beta_initializer = tf.zeros, gamma_initializer=tf.ones ,center=True,scale=True,name='batch_normalization_layer',epsilon=1e-6,decay=0.99):
    input_shape     = tensor.get_shape().as_list()
    shape_          = [s if i not in axis else 1 for i,s in enumerate(input_shape)]
    beta            = tf.Variable(beta_initializer(shape_),trainable=center,name=name+'_beta')
    gamma           = tf.Variable(gamma_initializer(shape_),trainable=scale,name=name+'_gamma')
    moving_mean     = tf.Variable(tf.zeros(shape_),trainable=False,name=name+'_movingmean')
    moving_var      = tf.Variable(tf.ones(shape_),trainable=False,name=name+'_movingvar')
    cpt             = tf.Variable(tf.ones(1),trainable=False,name=name+'cpt')
    moments         = tf.nn.moments(tensor,axes=axis,keep_dims=True)
    coeff           = (cpt-1.)/cpt
    update_mean     = tf.assign(moving_mean,tf.cond(training,lambda :moving_mean*decay+moments[0]*(1-decay),lambda :moving_mean))
    update_var      = tf.assign(moving_var,tf.cond(training,lambda :moving_var*decay+moments[1]*(1-decay),lambda :moving_var))
    update_cpt      = tf.assign_add(cpt,tf.cond(training,lambda :tf.ones(1),lambda :tf.zeros(1)))
    update_ops      = tf.group(update_mean,update_var,update_cpt)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,update_ops)
    bias    = -(gamma*tf.cond(training,lambda :moments[0],lambda :moving_mean))/tf.sqrt(tf.cond(training,lambda :moments[1],lambda :moving_var)+epsilon)+beta
    scaling = gamma/tf.sqrt(tf.cond(training,lambda :moments[1],lambda :moving_var)+epsilon)
    return scaling,bias


class Pool2DLayer:
    def __init__(self,incoming,window,pool_type='MAX'):
        self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]//window,incoming.output_shape[2]//window,incoming.output_shape[3])
        self.output = tf.nn.pool(incoming.output,(window,window),pool_type,padding='VALID',strides=(window,window))
        self.VQ     = None #TO CHANGE !
        print(self.output.get_shape().as_list())


class InputLayer:
    def __init__(self,input_shape,x):
        self.output       = x
        self.output_shape = input_shape
        print(self.output.get_shape().as_list())


class DenseLayer:
    def __init__(self,incoming,n_output,training,nonlinearity = tf.nn.relu,batch_norm = True,init_W = tf.contrib.layers.xavier_initializer(uniform=True),trainable=True):
        if(len(incoming.output_shape)>2):
            inputf = tf.layers.flatten(incoming.output)
            in_dim = prod(incoming.output_shape[1:])
        else:
            inputf = incoming.output
            in_dim = incoming.output_shape[1]
        self.output_shape = (incoming.output_shape[0],n_output)
        self.W            = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=trainable)
        Wx                = tf.matmul(inputf,self.W)
        if(batch_norm):  self.scaling,self.bias = batch_normalization(Wx,[0],training=training,center=trainable,scale=trainable)
        else:            self.scaling,self.bias = 1,tf.Variable(tf.zeros((1,n_output)),trainable=trainable,name='denselayer_b')
        self.S            = self.scaling*Wx+self.bias
        self.VQ           = tf.greater(self.S,0)
        if(nonlinearity is tf.nn.relu):
            self.mask = tf.cast(self.VQ,tf.float32)
        elif(nonlinearity is tf.abs):
            self.mask = tf.cast(self.VQ,tf.float32)*2-1
        elif(nonlinearity is tf.nn.leaky_relu):
            self.mask = tf.cast(self.VQ,tf.float32)*0.8+0.2
        elif(nonlinearity is tf.identity):
            #this is for the last layer case
            self.mask = 1.
            self.VQ   = tf.argmax(self.S,1)
        self.output   = self.S*self.mask
        print(self.output.get_shape().as_list())
#    def backward(self,output):
#        """output is of shape (batch,n_output)
#        return of this function is of shape [(batch,in_dim),(batch,1)]"""
#        # use tf.nn.conv2d_backprop_input for conv2D
#        A = tf.reshape(tf.matmul(output*self.mask*scaling,self.W,transpose_b=True),incoming.output_shape)
#        B = tf.matmul(output*self.mask,self.b,transpose_b=True)
#        return A,B


class ConstraintDenseLayer:
    def __init__(self,incoming,n_output,constraint='none',training=None):
        # bias_option : {unconstrained,constrained,zero}
        if(len(incoming.output_shape)>2): reshape_input = tf.layers.flatten(incoming.output)
        else:                             reshape_input = incoming.output
        in_dim      = prod(incoming.output_shape[1:])
        self.gamma  = tf.Variable(ones(1,float32),trainable=False)
        gamma_update= tf.assign(self.gamma,tf.clip_by_value(tf.cond(training,lambda :self.gamma*1.005,lambda :self.gamma),0,60000))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS,gamma_update)
        init_W      = tf.contrib.layers.xavier_initializer(uniform=True)
        if(constraint=='none'):
                self.W_     = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=True)
                self.W      = self.W_
        elif(constraint=='dt'):
                self.W_     = tf.Variable(init_W((in_dim,n_output)),name='W_dense',trainable=True)
                self.alpha  = tf.Variable(randn(1,n_output).astype('float32'),trainable=True)
                self.W      = self.alpha*tf.nn.softmax(tf.clip_by_value(self.gamma*self.W_,-20000,20000),axis=0)
        elif(constraint=='diag'):
                self.sign   = tf.Variable(randn(in_dim,n_output).astype('float32'),trainable=True)
                self.alpha  = tf.Variable((randn(1,n_output)/sqrt(n_output)).astype('float32'),trainable=True)
                self.W      = tf.nn.tanh(self.gamma*self.sign)*self.alpha
        self.output_shape = (incoming.output_shape[0],n_output)
        self.output       = tf.matmul(reshape_input,self.W)
        self.VQ = None


#class GlobalPoolLayer:
#    def __init__(self,incoming):
#        self.output       = tf.reduce_mean(incoming.output,[1,2],keep_dims=True)
#        self.output_shape = [incoming.output_shape[0],1,1,incoming.output_shape[3]]
#        self.VQ = None
#        print(self.output.get_shape().as_list())



class Conv2DLayer:
    def __init__(self,incoming,n_filters,filter_shape,training,nonlinearity = tf.nn.relu,batch_norm = True,init_W = tf.contrib.layers.xavier_initializer(uniform=True),stride=1,pad='valid',mode='CONSTANT',trainable=True):
        if(pad=='valid' or filter_shape==1):
            padded_input = incoming.output
            self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)//stride,(incoming.output_shape[1]-filter_shape+1)//stride,n_filters)
        elif(pad=='same'):
            assert(filter_shape%2 ==1)
            p = (filter_shape-1)/2
            padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
            self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]//stride,incoming.output_shape[2]//stride,n_filters)
        else:
            p = filter_shape-1
            padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
            self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)//stride,(incoming.output_shape[1]+filter_shape-1)//stride,n_filters)
        self.W      = tf.Variable(init_W((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=trainable)
        Wx                = tf.nn.conv2d(padded_input,self.W,strides=[1,stride,stride,1],padding='VALID')
        if(batch_norm):  self.scaling,self.bias = batch_normalization(Wx,[0,1,2],training=training,center=trainable,scale=trainable)
        else:            self.scaling,self.bias = 1,tf.Variable(tf.zeros((1,1,1,n_filters)),trainable=trainable,name='convlayer_b')
        self.S            = self.scaling*Wx+self.bias
        self.output       = nonlinearity(self.S)
        self.VQ           = tf.greater(self.output,0)
        W_norm            = tf.sqrt(tf.reduce_sum(tf.square(self.W*self.scaling),[0,1,2]))
        self.positive_radius = tf.reduce_min(tf.where(tf.greater(self.S,tf.zeros_like(self.S)),self.S,tf.reduce_max(self.S,keepdims=True)),[1,2,3])
        self.negative_radius = tf.reduce_min(tf.where(tf.smaller(self.S,tf.zeros_like(self.S)),tf.abs(self.S),tf.reduce_max(tf.abs(self.S),keepdims=True)),[1,2,3])
        print(self.output.get_shape().as_list())




class ConstrainConv2DLayer:
    def __init__(self,incoming,filters_T,channels_T,filter_shape,stride=1,pad='valid',mode='CONSTANT',spline=True):
        print(incoming.output_shape[3],(channels_T-1))
        sampling_n = int32(ceil(incoming.output_shape[3]/(channels_T-1)+1))
        n_filters  = (filters_T-1)*(sampling_n-1)
        print("sampling :{}\t n_filters :{}".format(sampling_n,n_filters))
        if(pad=='valid' or filter_shape==1):
            padded_input = incoming.output
            self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]-filter_shape+1)//stride,(incoming.output_shape[1]-filter_shape+1)//stride,n_filters)
        elif(pad=='same'):
            assert(filter_shape%2 ==1)
            p = (filter_shape-1)/2
            padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
            self.output_shape = (incoming.output_shape[0],incoming.output_shape[1]//stride,incoming.output_shape[2]//stride,n_filters)
        else:
            p = filter_shape-1
            padded_input = tf.pad(incoming.output,[[0,0],[p,p],[p,p],[0,0]],mode=mode)
            self.output_shape = (incoming.output_shape[0],(incoming.output_shape[1]+filter_shape-1)//stride,(incoming.output_shape[1]+filter_shape-1)//stride,n_filters)
        if(spline):
            self.spline = Spline2D(channels_T,filters_T,filter_shape,filter_shape)
            W           = self.spline.sample(sampling_n)
            self.W      = W[:,:,:incoming.output_shape[3]]
        else:
            init_W = tf.contrib.layers.xavier_initializer(uniform=True)
            self.W = tf.Variable(init_W((filter_shape,filter_shape,incoming.output_shape[3],n_filters)),name='W_conv2d',trainable=True)
        self.output = tf.nn.conv2d(padded_input,self.W,strides=[1,stride,stride,1],padding='VALID')
        self.VQ    = None
        print(self.output.get_shape().as_list())
        print(self.output_shape)













