import gzip
import io
import os
import time
import urllib.request
import zipfile

import numpy as np
import tqdm
from .utils import download_dataset


_urls = {"http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip": "gzip.zip"}


def _read_images(filename, folder):

    mat = folder.read(filename)
    f = gzip.open(io.BytesIO(mat), "rb")

    magic = f.read(4)
    magic = int.from_bytes(magic, "big")
    print("Magic is:", magic)

    # Number of images in next 4 bytes
    noimg = f.read(4)
    noimg = int.from_bytes(noimg, "big")

    # Number of rows in next 4 bytes
    norow = f.read(4)
    norow = int.from_bytes(norow, "big")

    # Number of columns in next 4 bytes
    nocol = f.read(4)
    nocol = int.from_bytes(nocol, "big")

    images = np.empty((noimg, norow, nocol), "float32")

    for i in tqdm.tqdm(range(noimg), ascii=True):
        for r in range(norow):
            for c in range(nocol):
                images[i, c, r] = float(int.from_bytes(f.read(1), "big"))

    f.close()

    return images


def _read_labels(filename, folder):

    mat = folder.read(filename)
    f = gzip.open(io.BytesIO(mat), "rb")

    magic = f.read(4)
    magic = int.from_bytes(magic, "big")
    print("Magic is:", magic)

    # Number of images in next 4 bytes
    nolab = f.read(4)
    nolab = int.from_bytes(nolab, "big")

    labels = [f.read(1) for i in range(nolab)]
    labels = [int.from_bytes(label, "big") for label in labels]

    f.close()

    return np.array(labels)


def load(option="byclass", path=None):
    """Grayscale digit/letter classification.

    The EMNIST Dataset

    Authors
    -------
    Gregory Cohen, Saeed Afshar, Jonathan Tapson,
    and Andre van Schaik

    The MARCS Institute for Brain, Behaviour and Development
    Western Sydney University
    Penrith, Australia 2751

    Email: g.cohen@westernsydney.edu.au

    What is it?
    -----------
    The EMNIST dataset is a set of handwritten character digits
    derived from the NIST Special Database 19
    (https://www.nist.gov/srd/nist-special-database-19) and
    converted to a 28x28 pixel image format and dataset structure
    that directly matches the MNIST dataset
    (http://yann.lecun.com/exdb/mnist/). Further information on
    the dataset contents and conversion process can be found in
    the paper available at https://arxiv.org/abs/1702.05373v1.

    Formats
    -------
    The dataset is provided in two file formats. Both versions of
    the dataset contain identical information, and are provided
    entirely for the sake of convenience. The first dataset is
    provided in a Matlab format that is accessible through both
    Matlab and Python (using the scipy.io.loadmat function). The
    second version of the dataset is provided in the same binary
    format as the original MNIST dataset as outlined in
    http://yann.lecun.com/exdb/mnist/

    Dataset Summary
    ---------------
    There are six different splits provided in this dataset.
    A short summary of the dataset is provided below:

    EMNIST ByClass:EMNIST814,255 characters. 62 unbalanced classes
    EMNIST ByMerge:     814,255 characters. 47 unbalanced classes
    EMNIST Balanced:Balanced131,600 characters. 47 balanced classes.
    EMNIST Letters:EMNIST145,600 characters. 26 balanced classes.
    EMNIST Digits:EMNIST280,000 characters. 10 balanced classes.
    EMNIST MNIST:EMNIST 70,000 characters. 10 balanced classes.

    The full complement of the NIST Special Database 19 is
    available in the ByClass and ByMerge splits. The EMNIST
    Balanced dataset contains a set of characters with an equal
    number of samples per class. The EMNIST Letters dataset
    merges a balanced set of the uppercase and lowercase letters
    into a single 26-class task. The EMNIST Digits and EMNIST
    MNIST dataset provide balanced handwritten digit datasets
    directly compatible with the original MNIST dataset.

    Please refer to the EMNIST paper (available at
    https://arxiv.org/abs/1702.05373v1) for further details of
    the dataset structure.

    How to cite
    -----------
    Please cite the following paper when using or referencing
    the dataset:

    Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017).
    EMNIST: an extension of MNIST to handwritten letters.
    Retrieved from http://arxiv.org/abs/1702.05373

    Files
    -----
    The dataset consists of the following files:

    .
    +-- gzip.zip
    ¦   +-- emnist-balanced-mapping.txt
    ¦   +-- emnist-balanced-test-images-idx3-ubyte.gz
    ¦   +-- emnist-balanced-test-labels-idx1-ubyte.gz
    ¦   +-- emnist-balanced-train-images-idx3-ubyte.gz
    ¦   +-- emnist-balanced-train-labels-idx1-ubyte.gz
    ¦   +-- emnist-byclass-mapping.txt
    ¦   +-- emnist-byclass-test-images-idx3-ubyte.gz
    ¦   +-- emnist-byclass-test-labels-idx1-ubyte.gz
    ¦   +-- emnist-byclass-train-images-idx3-ubyte.gz
    ¦   +-- emnist-byclass-train-labels-idx1-ubyte.gz
    ¦   +-- emnist-bymerge-mapping.txt
    ¦   +-- emnist-bymerge-test-images-idx3-ubyte.gz
    ¦   +-- emnist-bymerge-test-labels-idx1-ubyte.gz
    ¦   +-- emnist-bymerge-train-images-idx3-ubyte.gz
    ¦   +-- emnist-bymerge-train-labels-idx1-ubyte.gz
    ¦   +-- emnist-digits-mapping.txt
    ¦   +-- emnist-digits-test-images-idx3-ubyte.gz
    ¦   +-- emnist-digits-test-labels-idx1-ubyte.gz
    ¦   +-- emnist-digits-train-images-idx3-ubyte.gz
    ¦   +-- emnist-digits-train-labels-idx1-ubyte.gz
    ¦   +-- emnist-letters-mapping.txt
    ¦   +-- emnist-letters-test-images-idx3-ubyte.gz
    ¦   +-- emnist-letters-test-labels-idx1-ubyte.gz
    ¦   +-- emnist-letters-train-images-idx3-ubyte.gz
    ¦   +-- emnist-letters-train-labels-idx1-ubyte.gz
    ¦   +-- emnist-mnist-mapping.txt
    ¦   +-- emnist-mnist-test-images-idx3-ubyte.gz
    ¦   +-- emnist-mnist-test-labels-idx1-ubyte.gz
    ¦   +-- emnist-mnist-train-images-idx3-ubyte.gz
    ¦   +-- emnist-mnist-train-labels-idx1-ubyte.gz
    +-- matlab.zip
        +-- emnist-balanced.mat
            +-- emnist-byclass.mat
                +-- emnist-bymerge.mat
                    +-- emnist-digits.mat
                        +-- emnist-letters.mat
                            +-- emnist-mnist.mat
                            +-- Readme.txt

    """
    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, "EMNIST", _urls)

    # Loading the file
    print("Loading emnist")
    folder = zipfile.ZipFile(path + "EMNIST/gzip.zip")
    filename = "gzip/emnist-{}-{}-{}-idx{}-ubyte.gz"
    x_test = _read_images(filename.format(option, "test", "images", 3), folder)
    x_train = _read_images(filename.format(option, "train", "images", 3), folder)
    y_test = _read_labels(filename.format(option, "test", "labels", 1), folder)
    y_train = _read_labels(filename.format(option, "train", "labels", 1), folder)
    data = {
        "train_set/images": x_train,
        "train_set/labels": y_train,
        "test_set/images": x_test,
        "test_set/labels": y_test,
    }
    return data
