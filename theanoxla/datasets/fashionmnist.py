import os
import gzip
import urllib.request
import numpy as np
import time

from . import Dataset

from ..utils import to_one_hot, DownloadProgressBar



def load_fashionmnist(PATH=None):
    """Grayscale `Zalando <https://jobs.zalando.com/tech/>`_ 's article image classification.
    `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ is 
    a dataset of `Zalando <https://jobs.zalando.com/tech/>`_ 's article 
    images consisting of a training set of 60,000 examples and a test set 
    of 10,000 examples. Each example is a 28x28 grayscale image, associated 
    with a label from 10 classes. We intend Fashion-MNIST to serve as a direct
    drop-in replacement for the original MNIST dataset for benchmarking 
    machine learning algorithms. It shares the same image size and structure 
    of training and testing splits.

    Parameters
    ----------
        path: str (optional)
            default $DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present
    """
    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    dict_init = [("n_classes",10),("path",PATH),("name","fashionmnist"),
                ("classes",["T-shirt/top", "Trouser", "Pullover",
                    "Dress", "Coat", "Sandal", "Shirt",
                    "Sneaker", "Bag", "Ankle boot"])]

    dataset = Dataset(**dict(dict_init))
    # Load the dataset (download if necessary) and set
    # the class attributes.
    print('Loading fashionmnist')

    t0 = time.time()

    if not os.path.isdir(PATH+'fashionmnist'):
        print('\tCreating Directory')
        os.mkdir(PATH+'fashionmnist')
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'
    if not os.path.exists(PATH+'fashionmnist/train-images.gz'):
        url = base_url+'train-images-idx3-ubyte.gz'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                        desc='Downloading train images') as t:
            urllib.request.urlretrieve(url,PATH+'fashionmnist/train-images.gz')

    if not os.path.exists(PATH+'fashionmnist/train-labels.gz'):
        url = base_url+'train-labels-idx1-ubyte.gz'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                        desc='Downloading train labels') as t:
            urllib.request.urlretrieve(url,PATH+'fashionmnist/train-labels.gz')

    if not os.path.exists(PATH+'fashionmnist/test-images.gz'):
        url = base_url+'t10k-images-idx3-ubyte.gz'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                        desc='Downloading test images') as t:
            urllib.request.urlretrieve(url,PATH+'fashionmnist/test-images.gz')

    if not os.path.exists(PATH+'fashionmnist/test-labels.gz'):
        url = base_url+'t10k-labels-idx1-ubyte.gz'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                        desc='Downloading test labels') as t:
            urllib.request.urlretrieve(url,PATH+'fashionmnist/test-labels.gz')

    # Loading the file

    with gzip.open(PATH+'fashionmnist/train-labels.gz', 'rb') as lbpath:
        dataset['labels/train_set'] = np.frombuffer(lbpath.read(), 
                                                   dtype=np.uint8, offset=8)

    with gzip.open(PATH+'fashionmnist/train-images.gz', 'rb') as lbpath:
        dataset['images/train_set'] = np.frombuffer(lbpath.read(), 
                                dtype=np.uint8, offset=16).reshape((-1,1,28,28))

    with gzip.open(PATH+'fashionmnist/test-labels.gz', 'rb') as lbpath:
        dataset['labels/test_set'] = np.frombuffer(lbpath.read(), 
                                                   dtype=np.uint8, offset=8)

    with gzip.open(PATH+'fashionmnist/test-images.gz', 'rb') as lbpath:
        dataset['images/test_set'] = np.frombuffer(lbpath.read(), 
                            dtype=np.uint8, offset=16).reshape((-1,1,28,28))

    dataset.cast('images','float32')
    dataset.cast('labels','int32')

    print('Dataset fashionmnist loaded in','{0:.2f}s.'.format(time.time()-t0))
    return dataset
