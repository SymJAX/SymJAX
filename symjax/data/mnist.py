import os
import pickle, gzip
import urllib.request
import numpy as np
import time


def download(path):
    """
    Download the MNIST dataset and store the result into the given
    path

    Parameters
    ----------

        path: str
            the path where the downloaded files will be stored. If the
            directory does not exist, it is created.
    """

    # Check if directory exists
    if not os.path.isdir(path + "mnist"):
        print("Creating mnist Directory")
        os.mkdir(path + "mnist")

    # Check if file exists
    if not os.path.exists(path + "mnist/mnist.pkl.gz"):
        td = time.time()
        print("Creating mnist")
        url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
        urllib.request.urlretrieve(url, path + "mnist/mnist.pkl.gz")


def load(path=None):
    """
    Parameters
    ----------
        path: str (optional)
            default ($DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present

    Returns
    -------

        train_images: array

        train_labels: array

        valid_images: array

        valid_labels: array

        test_images: array

        test_labels: array

    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download(path)

    t0 = time.time()

    # Loading the file
    print("Loading mnist")
    f = gzip.open(path + "mnist/mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    f.close()

    train_set = (
        train_set[0].reshape((-1, 1, 28, 28)).astype("float32"),
        train_set[1].astype("int32"),
    )
    test_set = (
        test_set[0].reshape((-1, 1, 28, 28)).astype("float32"),
        test_set[1].astype("int32"),
    )
    valid_set = (
        valid_set[0].reshape((-1, 1, 28, 28)).astype("float32"),
        valid_set[1].astype("int32"),
    )

    data = {
        "train_set/images": train_set[0],
        "train_set/labels": train_set[1],
        "test_set/images": test_set[0],
        "test_set/labels": test_set[1],
        "valid_set/images": valid_set[0],
        "valid_set/labels": valid_set[1],
    }

    print("Dataset mnist loaded in {0:.2f}s.".format(time.time() - t0))

    return data
