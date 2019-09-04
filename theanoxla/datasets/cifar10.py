import os
import pickle,gzip
import urllib.request
import numpy as np
import tarfile
import time
from tqdm import tqdm

from . import Dataset

from ..utils import to_one_hot, DownloadProgressBar


def load_cifar10(PATH=None):
    """Image classification.
    The `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset 
    was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey 
    Hinton. It consists of 60000 32x32 colour images in 10 classes, with 
    6000 images per class. There are 50000 training images and 10000 test images. 
    The dataset is divided into five training batches and one test batch, 
    each with 10000 images. The test batch contains exactly 1000 randomly
    selected images from each class. The training batches contain the 
    remaining images in random order, but some training batches may 
    contain more images from one class than another. Between them, the 
    training batches contain exactly 5000 images from each class. 

    Parameters
    ----------
        path: str (optional)
            default $DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present

    """
    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    dict_init = [("n_classes",10),("path",PATH),("name","cifar10")]
    class_init = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                "frog", "horse", "ship", "truck"]
    dataset = Dataset(**dict(dict_init+[('classes',class_init)]))

    # Load the dataset (download if necessary) and set
    # the class attributes.
        
    t0 = time.time()

    print('Loading cifar10')

    # Check if directory exists
    if not os.path.isdir(PATH+'cifar10'):
        print('\tCreating Directory')
        os.mkdir(PATH+'cifar10')

    # Check if file exists
    if not os.path.exists(PATH+'cifar10/cifar10.tar.gz'):
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                            desc='Downloading Dataset') as t:
            urllib.request.urlretrieve(url,PATH+'cifar10/cifar10.tar.gz')

    # Loading dataset
    tar = tarfile.open(PATH+'cifar10/cifar10.tar.gz', 'r:gz')

    # Load train set
    train_images  = list()
    train_labels  = list()
    for k in tqdm(range(1,6),desc='Loading dataset'):
        f        = tar.extractfile(
                        'cifar-10-batches-py/data_batch_'+str(k)).read()
        data_dic = pickle.loads(f,encoding='latin1')
        train_images.append(data_dic['data'].reshape((-1,3,32,32)))
        train_labels.append(data_dic['labels'])
    dataset['images/train_set'] = np.concatenate(train_images,0)
    dataset['labels/train_set'] = np.concatenate(train_labels,0)

    # Load test set
    f        = tar.extractfile('cifar-10-batches-py/test_batch').read()
    data_dic = pickle.loads(f,encoding='latin1')
    dataset['images/test_set'] = data_dic['data'].reshape((-1,3,32,32))
    dataset['labels/test_set'] = np.array(data_dic['labels'])

    dataset.cast('images','float32')
    dataset.cast('labels','int32')

    print('Dataset cifar10 loaded in{0:.2f}s.'.format(time.time()-t0))

    return dataset
