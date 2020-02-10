import tensorflow as tf
import pickle
from pylab import *
import glob
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split





def count_number_of_params():
    return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])


class Spline2D:
    def __init__(self,N,M,n_x,n_y):
        #N and M is the number of knots
        # R_N,R_M is the number of region
        self.R_N = N-1
        self.R_M = M-1
        self.n_x = n_x
        self.n_y = n_y
        VALUES   = tf.Variable(randn(n_x*n_y,N,M,4).astype('float32')/(N*M))#values vx vy vxy
        patches  = tf.reshape(tf.extract_image_patches(VALUES,ksizes=[1,2,2,1],strides=[1,1,1,1],rates=[1,1,1,1],padding="VALID"),[n_x*n_y,self.R_N,self.R_M,2,2,4]) #(n_x*n_y,R_N,R_M,2,2,4)
        self.F   = tf.concat([tf.concat([patches[...,0],patches[...,1]],-1),tf.concat([patches[...,2],patches[...,3]],-1)],axis=-2) #(n_x*n_y,R_N,R_M,4,4)
        self.M   = tf.Variable(array([[1,0,0,0],[0,0,1,0],[-3,3,-2,-1],[2,-2,1,1]]).astype('float32'),trainable=False) # (4 4)
        self.MFM = tf.einsum('xnmij,ai,bj->xnmab',self.F,self.M,self.M) # (n_x*n_y,R_N,R_M,4,4)
    def sample(self,n):
        #filter shape is (n_x,n_y,(n-1)*self.R_N,(n-1)*self.R_M)
        t      = tf.lin_space(float32(0),float32(1),int32(n-1))
        x      = tf.stack([tf.ones(n-1),t,tf.square(t),tf.pow(t,3)])  # (4,n-1)
        values = tf.einsum('xnmij,ia,jb->xnamb',self.MFM,x,x) # (n_x*n_y,R_N,n-1,R_M,n-1)
        return tf.reshape(values,(self.n_x,self.n_y,(self.R_N)*(n-1),(self.R_M)*(n-1)))


###################################################################
#
#
#                       UTILITY FOR DATASET LOADING
#
#
###################################################################


def load_utility(DATASET):
    if(DATASET=='MNIST'):
        batch_size = 50
        mnist         = fetch_mldata('MNIST original')
        x             = mnist.data.reshape(70000,1,28,28).astype('float32')
        y             = mnist.target.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=10000,stratify=y)
        input_shape   = (batch_size,28,28,1)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
        c = 10
        n_epochs = 150

    elif(DATASET == 'CIFAR'):
        batch_size = 50
        TRAIN,TEST = load_cifar(3)
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
        c=10
        n_epochs = 150

    elif(DATASET == 'CIFAR100'):
        batch_size = 100
        TRAIN,TEST = load_cifar100(3)
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
        c=100
        n_epochs = 200

    elif(DATASET=='IMAGE'):
        batch_size=200
        x,y           = load_imagenet()
        x = x.astype('float32')
        y = y.astype('int32')
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=20000,stratify=y)
        input_shape   = (batch_size,64,64,3)
        c=200
        n_epochs = 200

    else:
        batch_size = 50
        TRAIN,TEST = load_svhn()
        x_train,y_train = TRAIN
        x_test,y_test     = TEST
        input_shape       = (batch_size,32,32,3)
        x_train = transpose(x_train,[0,2,3,1])
        x_test  = transpose(x_test,[0,2,3,1])
        c=10
        n_epochs = 150

    x_train          -= x_train.mean((1,2,3),keepdims=True)
    x_train          /= abs(x_train).max((1,2,3),keepdims=True)
    x_test           -= x_test.mean((1,2,3),keepdims=True)
    x_test           /= abs(x_test).max((1,2,3),keepdims=True)
    x_train           = x_train.astype('float32')
    x_test            = x_test.astype('float32')
    y_train           = array(y_train).astype('int32')
    y_test            = array(y_test).astype('int32')
    return x_train,x_test,y_train,y_test,c,n_epochs,input_shape 


def principal_components(x):
    x = x.transpose(0, 2, 3, 1)
    flatx = numpy.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))
    sigma = numpy.dot(flatx.T, flatx) / flatx.shape[1]
    U, S, V = numpy.linalg.svd(sigma)
    eps = 0.0001
    return numpy.dot(numpy.dot(U, numpy.diag(1. / numpy.sqrt(S + eps))), U.T)


def zca_whitening(x, principal_components):
#    x = x.transpose(1,2,0)
    flatx = numpy.reshape(x, (x.size))
    whitex = numpy.dot(flatx, principal_components)
    x = numpy.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))
    return x

def load_imagenet():
    import scipy.misc
    classes = glob.glob('../../DATASET/tiny-imagenet-200/train/*')
    x_train,y_train = [],[]
    cpt=0
    for c,name in zip(range(200),classes):
        files = glob.glob(name+'/images/*.JPEG')
        for f in files:
            x_train.append(scipy.misc.imread(f, flatten=False, mode='RGB'))
            y_train.append(c)
    return asarray(x_train),asarray(y_train)



def load_svhn():
    import scipy.io as sio
    train_data = sio.loadmat('../../DATASET/train_32x32.mat')
    x_train = train_data['X'].transpose([3,2,0,1]).astype('float32')
    y_train = concatenate(train_data['y']).astype('int32')-1
    test_data = sio.loadmat('../../DATASET/test_32x32.mat')
    x_test = test_data['X'].transpose([3,2,0,1]).astype('float32')
    y_test = concatenate(test_data['y']).astype('int32')-1
    return [x_train,y_train],[x_test,y_test]



def unpickle100(file,labels,channels):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    if(channels==1):
        p=dict['data'][:,:1024]*0.299+dict['data'][:,1024:2048]*0.587+dict['data'][:,2048:]*0.114
        p = p.reshape((-1,1,32,32))#dict['data'].reshape((-1,3,32,32))
    else:
        p=dict['data']
        p = p.reshape((-1,channels,32,32)).astype('float64')#dict['data'].reshape((-1,3,32,32))
    if(labels == 0 ):
        return p
    else:
        return asarray(p),asarray(dict['fine_labels'])



def unpickle(file,labels,channels):
    fo = open(file, 'rb')
    dict = pickle.load(fo,encoding='latin1')
    fo.close()
    if(channels==1):
        p=dict['data'][:,:1024]*0.299+dict['data'][:,1024:2048]*0.587+dict['data'][:,2048:]*0.114
        p = p.reshape((-1,1,32,32))#dict['data'].reshape((-1,3,32,32))
    else:
        p=dict['data']
        p = p.reshape((-1,channels,32,32)).astype('float64')#dict['data'].reshape((-1,3,32,32))
    if(labels == 0 ):
        return p
    else:
        return asarray(p),asarray(dict['labels'])





def load_mnist():
    mndata = file('../DATASET/MNIST.pkl','rb')
    data=pickle.load(mndata)
    mndata.close()
    return [concatenate([data[0][0],data[1][0]]).reshape(60000,1,28,28),concatenate([data[0][1],data[1][1]])],[data[2][0].reshape(10000,1,28,28),data[2][1]]

def load_cifar(channels=1):
    path = '../../DATASET/cifar-10-batches-py/'
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']:
        PP = unpickle(path+i,1,channels)
        x_train.append(PP[0])
        y_train.append(PP[1])
    x_test,y_test = unpickle(path+'test_batch',1,channels)
    x_train = concatenate(x_train)
    y_train = concatenate(y_train)
    return [x_train,y_train],[x_test,y_test]



def load_cifar100(channels=1):
    path = '../../DATASET/cifar-100-python/'
    PP = unpickle100(path+'train',1,channels)
    x_train = PP[0]
    y_train = PP[1]
    PP = unpickle100(path+'test',1,channels)
    x_test = PP[0]
    y_test = PP[1]
    return [x_train,y_train],[x_test,y_test]











