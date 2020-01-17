import os
import pickle,gzip
import urllib.request
import numpy as np
import time

#from ..utils import  DownloadProgressBar

def download(path):
    # Check if directory exists
    if not os.path.isdir(path+'mnist'):
        print('Creating mnist Directory')
        os.mkdir(path+'mnist')

    # Check if file exists
    if not os.path.exists(path+'mnist/mnist.pkl.gz'):
        td  = time.time()
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
#        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
#                                                    desc='DL dataset') as t:
        urllib.request.urlretrieve(url,path + 'mnist/mnist.pkl.gz')



def load(PATH=None, classes=range(10)):
    """Grayscale digit classification.
    The `MNIST <http://yann.lecun.com/exdb/mnist/>`_ database of handwritten 
    digits, available from this page, has a training set of 60,000 examples, 
    and a test set of 10,000 examples. It is a subset of a larger set available 
    from NIST. The digits have been size-normalized and centered in a 
    fixed-size image. It is a good database for people who want to try learning
    techniques and pattern recognition methods on real-world data while 
    spending minimal efforts on preprocessing and formatting.

    Parameters
    ----------
        path: str (optional)
            default $DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present
    """

    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    download(PATH)

    t0 = time.time()

    # Loading the file
    f = gzip.open(PATH+'mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
    f.close()

    train_set = (train_set[0].reshape((-1,1,28,28)).astype('float32'),
                 train_set[1].astype('int32'))
    test_set = (test_set[0].reshape((-1,1,28,28)).astype('float32'),
                 test_set[1].astype('int32'))
    valid_set = (valid_set[0].reshape((-1,1,28,28)).astype('float32'),
                 valid_set[1].astype('int32'))


    print('Dataset mnist loaded in {0:.2f}s.'.format(time.time()-t0))

    return train_set, valid_set, test_set

