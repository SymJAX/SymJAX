import urllib.request
import time
import sys
import os, sys, tarfile, io
import numpy as np
import matplotlib.pyplot as plt


class stl10:
    """ Image classification with extra unlabeled images.

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

    """

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

    @staticmethod
    def download(path):
        # Check if directory exists
        if not os.path.isdir(path + "stl10"):
            print("\tCreating stl10 Directory")
            os.mkdir(path + "stl10")

        # Check if data file exists
        if not os.path.exists(path + "stl10/stl10_binary.tar.gz"):
            print("\tDownloading stl10 Dataset...")
            url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
            urllib.request.urlretrieve(url, path + "stl10/stl10_binary.tar.gz")

    @staticmethod
    def load(path=None):
        """
        load the data

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
            path = os.environ["DATASET_path"]

        # Load the dataset (download if necessary) and set
        # the class attributes.

        print("Loading stl10")
        t = time.time()

        stl10.download(path)

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
        return train_X, train_y, test_X, test_y, unlabeled_X
