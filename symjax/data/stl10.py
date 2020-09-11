import time
import os, tarfile, io
import numpy as np
from .utils import download_dataset


_urls = {
    "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz": "stl10_binary.tar.gz",
}


classes = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]


def load(path=None):
    """Image classification with extra unlabeled images.

    The `STL-10 <https://cs.stanford.edu/~acoates/stl10/>`_ dataset is an image
    recognition dataset for developing unsupervised feature learning,
    deep learning, self-taught learning algorithms. It is inspired by the
    CIFAR-10 dataset but with
    some modifications. In particular, each class has fewer labeled
    training examples than in CIFAR-10, but a very
    large set of unlabeled examples is provided to learn image models prior
    to supervised training. The primary challenge is to make use of the
    unlabeled data (which comes from a similar but different distribution from
    the labeled data) to build a useful prior. We also expect that the higher
    resolution of this dataset (96x96) will make it a challenging benchmark
    for developing more scalable unsupervised learning methods.


    Parameters
    ----------

    path: str (optional)
        the path to look for the data and where it will be downloaded if
        not present

    Returns
    -------

    train_images: array
        the training images

    train_labels: array
        the training labels

    test_images: array
        the test images

    test_labels: array
        the test labels

    extra_images: array
        the unlabeled additional images
    """
    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, "irmas", _urls)

    print("Loading stl10")
    t = time.time()

    # Loading Dataset
    file_ = tarfile.open(path + "stl10/stl10_binary.tar.gz", "r:gz")
    # loading test label
    read_file = file_.extractfile("stl10_binary/test_y.bin").read()
    test_y = np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8) - 1
    # loading train label
    read_file = file_.extractfile("stl10_binary/train_y.bin").read()
    train_y = np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8) - 1
    # load test images
    read_file = file_.extractfile("stl10_binary/test_X.bin").read()
    test_X = (
        np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8)
        .reshape((-1, 3, 96, 96))
        .transpose([0, 1, 3, 2])
    )
    # load train images
    read_file = file_.extractfile("stl10_binary/train_X.bin").read()
    train_X = (
        np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8)
        .reshape((-1, 3, 96, 96))
        .transpose([0, 1, 3, 2])
    )
    # load unlabelled images
    read_file = file_.extractfile("stl10_binary/unlabeled_X.bin").read()
    unlabeled_X = (
        np.frombuffer(io.BytesIO(read_file).read(), dtype=np.uint8)
        .reshape((-1, 3, 96, 96))
        .transpose([0, 1, 3, 2])
    )

    print("Dataset stl10 loaded in", "{0:.2f}".format(time.time() - t), "s.")
    data = {
        "train_set/images": train_X,
        "train_set/labels": train_y,
        "test_set/images": test_X,
        "test_set/labels": test_y,
        "unlabelled": unlabeled_X,
    }
    return data
