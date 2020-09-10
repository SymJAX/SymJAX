import os
import gzip
import urllib.request
import numpy as np
import time
from .utils import download_dataset


_urls = {
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz": "train-images.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz": "train-labels.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz": "test-images.gz",
    "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz": "test-labels.gz",
}


def load(path=None):
    """Grayscale image classification

    `Zalando <https://jobs.zalando.com/tech/>`_ 's article image classification.
    `Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ is
    a dataset of `Zalando <https://jobs.zalando.com/tech/>`_ 's article
    images consisting of a training set of 60,000 examples and a test set
    of 10,000 examples. Each example is a 28x28 grayscale image, associated
    with a label from 10 classes. We intend Fashion-MNIST to serve as a direct
    drop-in replacement for the original MNIST dataset for benchmarking
    machine learning algorithms. It shares the same image size and structure
    of training and testing splits.
    """
    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, "fashionmnist", _urls)

    t0 = time.time()
    print("\tLoading fashionmnist")
    with gzip.open(path + "fashionmnist/train-labels.gz", "rb") as lbpath:
        train_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(path + "fashionmnist/train-images.gz", "rb") as lbpath:
        train_images = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16)
    train_images = train_images.reshape((-1, 1, 28, 28)).astype("float32")

    with gzip.open(path + "fashionmnist/test-labels.gz", "rb") as lbpath:
        test_labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(path + "fashionmnist/test-images.gz", "rb") as lbpath:
        test_images = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=16)
    test_images = test_images.reshape((-1, 1, 28, 28)).astype("float32")

    data = {
        "train_set/images": train_images,
        "train_set/labels": train_labels,
        "test_set/images": test_images,
        "test_set/labels": test_labels,
    }

    print("Dataset mnist loaded in {0:.2f}s.".format(time.time() - t0))

    return data
