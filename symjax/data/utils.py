import pickle
from pylab import *
import glob




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





