import os
import gzip
import urllib.request
import numpy as np
import time


class fashionmnist:
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

    def download(path):
        """
        Download the fashion-MNIST dataset and store the result into the given
        path

        Parameters
        ----------

            path: str
                the path where the downloaded files will be stored. If the
                directory does not exist, it is created.
        """

        print("Loading fashionmnist")

        t0 = time.time()

        if not os.path.isdir(path + "fashionmnist"):
            print("\tCreating fashionmnist Directory")
            os.mkdir(path + "fashionmnist")
        base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        if not os.path.exists(path + "fashionmnist/train-images.gz"):
            print("\tDownloading train images")
            url = base_url + "train-images-idx3-ubyte.gz"
            urllib.request.urlretrieve(url, path + "fashionmnist/train-images.gz")

        if not os.path.exists(path + "fashionmnist/train-labels.gz"):
            print("\tDownloading train labels")
            url = base_url + "train-labels-idx1-ubyte.gz"
            urllib.request.urlretrieve(url, path + "fashionmnist/train-labels.gz")

        if not os.path.exists(path + "fashionmnist/test-images.gz"):
            print("\tDownloading test images")
            url = base_url + "t10k-images-idx3-ubyte.gz"
            urllib.request.urlretrieve(url, path + "fashionmnist/test-images.gz")

        if not os.path.exists(path + "fashionmnist/test-labels.gz"):
            print("\tDownloading test labels")
            url = base_url + "t10k-labels-idx1-ubyte.gz"
            urllib.request.urlretrieve(url, path + "fashionmnist/test-labels.gz")

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

            test_images: array

            test_labels: array

        """
        if path is None:
            path = os.environ["DATASET_PATH"]
        # Loading the file
        fashionmnist.download(path)
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

        print("Dataset fashionmnist loaded in", "{0:.2f}s.".format(time.time() - t0))
        return (
            np.copy(train_images),
            np.copy(train_labels),
            np.copy(test_images),
            np.copy(test_labels),
        )
