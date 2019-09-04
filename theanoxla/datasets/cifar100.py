import urllib.request
import numpy as np
import tarfile
import os
import pickle
import time

from . import Dataset
from ..utils import to_one_hot, DownloadProgressBar


from . import Dataset

labels_list = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
    'worm'
]


def load_cifar100(PATH=None):
    """Image classification.
    The `CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset is 
    just like the CIFAR-10, except it has 100 classes containing 600 images 
    each. There are 500 training images and 100 testing images per class. 
    The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each 
    image comes with a "fine" label (the class to which it belongs) and a 
    "coarse" label (the superclass to which it belongs).

    Parameters
    ----------
        path: str (optional)
            default $DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present

    """

    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    dict_init = [("n_classes",100),("path",PATH),("name","cifar100"),
                ("classes",labels_list),("n_coarse_classes",20)]
    dataset = Dataset(**dict(dict_init))
    
    # Load the dataset (download if necessary) and set
    # the class attributes.
        
    print('Loading cifar100')
                
    t0 = time.time()

    if not os.path.isdir(PATH+'cifar100'):
        print('\tCreating cifar100 Directory')
        os.mkdir(PATH+'cifar100')

    if not os.path.exists(PATH+'cifar100/cifar100.tar.gz'):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                        desc='Downloading dataset') as t:
            urllib.request.urlretrieve(url,PATH+'cifar100/cifar100.tar.gz')

    # Loading the file
    tar = tarfile.open(PATH+'cifar100/cifar100.tar.gz', 'r:gz')

    # Loading training set
    f    = tar.extractfile('cifar-100-python/train').read()
    data = pickle.loads(f,encoding='latin1')
    dataset['images/train_set']        = data['data'].reshape((-1,3,32,32))
    dataset['labels/train_set']        = np.array(data['fine_labels'])
    dataset['coarse_labels/train_set'] = np.array(data['coarse_labels'])

    # Loading test set
    f    = tar.extractfile('cifar-100-python/test').read()
    data = pickle.loads(f,encoding='latin1')
    dataset['images/test_set']        = data['data'].reshape((-1,3,32,32))
    dataset['labels/test_set']        = np.array(data['fine_labels'])
    dataset['coarse_labels/test_set'] = np.array(data['coarse_labels'])

    dataset.cast('images','float32')
    dataset.cast('labels','int32')

    print('Dataset cifar100 loaded in {0:.2f}s.'.format(time.time()-t0))
    return dataset
