import os
import pickle,gzip
import urllib.request
import numpy as np
import time
from . import Dataset

from ..utils import to_one_hot, DownloadProgressBar


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
    #init
    #Set up the configuration for data loading and data format

    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    datum_shape = (1,28,28)
    dict_init = [("n_classes", len(classes)),("name", "mnist"),
                    ('classes', [str(u) for u in range(10)])]

    dataset = Dataset(**dict(dict_init))
    print('Loading mnist')

    t0 = time.time()

    # Check if directory exists
    if not os.path.isdir(PATH+'mnist'):
        print('Creating mnist Directory')
        os.mkdir(PATH+'mnist')

    # Check if file exists
    if not os.path.exists(PATH+'mnist/mnist.pkl.gz'):
        td  = time.time()
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                                    desc='DL dataset') as t:
            urllib.request.urlretrieve(url,PATH+'mnist/mnist.pkl.gz')

    # Loading the file
    f = gzip.open(PATH+'mnist/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
    f.close()

    itrain = np.isin(train_set[1], classes)
    itest = np.isin(test_set[1], classes)
    ivalid = np.isin(valid_set[1], classes)

    dataset['images/train_set'] = train_set[0][itrain].reshape((-1,1,28,28))
    dataset['images/test_set']  = test_set[0][itest].reshape((-1,1,28,28))
    dataset['images/valid_set'] = valid_set[0][ivalid].reshape((-1,1,28,28))

    dataset['labels/train_set'] = train_set[1][itrain]
    dataset['labels/test_set']  = test_set[1][itest]
    dataset['labels/valid_set'] = valid_set[1][ivalid]

    dataset.cast('images','float32')
    dataset.cast('labels','int32')

    print('Dataset mnist loaded in {0:.2f}s.'.format(time.time()-t0))
    return dataset

