import os
import pickle
import tarfile
import time
from .utils import download_dataset

import numpy as np
from tqdm import tqdm


_dataset = "cifar10"
_urls = {
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz": "cifar-10-python.tar.gz"
}


label_to_name = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "sheep",
    9: "truck",
}


def load(path=None):
    """Image classification.
    The `CIFAR-10 < https: // www.cs.toronto.edu/~kriz/cifar.html >`_ dataset
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
            default ($DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present

    Returns
    -------

        train_images: array

        train_labels: array

        test_images: array

        test_labels: array

    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, _dataset, _urls)

    t0 = time.time()

    tar = tarfile.open(os.path.join(path, _dataset, "cifar10.tar.gz"), "r:gz")

    # Load train set
    train_images = list()
    train_labels = list()
    for k in tqdm(range(1, 6), desc="Loading cifar10", ascii=True):
        f = tar.extractfile("cifar-10-batches-py/data_batch_" + str(k)).read()
        data_dic = pickle.loads(f, encoding="latin1")
        train_images.append(data_dic["data"].reshape((-1, 3, 32, 32)))
        train_labels.append(data_dic["labels"])
    train_images = np.concatenate(train_images, 0).astype("float32")
    train_labels = np.concatenate(train_labels, 0).astype("int32")

    # Load test set
    f = tar.extractfile("cifar-10-batches-py/test_batch").read()
    data_dic = pickle.loads(f, encoding="latin1")
    test_images = data_dic["data"].reshape((-1, 3, 32, 32)).astype("float32")
    test_labels = np.array(data_dic["labels"]).astype("int32")

    data = {
        "train_set/images": train_images,
        "train_set/labels": train_labels,
        "test_set/images": test_images,
        "test_set/labels": test_labels,
        "label_to_name": label_to_name,
    }

    print("Dataset cifar10 loaded in{0:.2f}s.".format(time.time() - t0))

    return data
