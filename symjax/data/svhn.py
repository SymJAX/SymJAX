import scipy.io as sio
import os
import pickle, gzip
import urllib.request
import numpy as np
import time


class svhn:
    """Street number classification.

    The `SVHN <http://ufldl.stanford.edu/housenumbers/>`_
    dataset is a real-world 
    image dataset for developing machine learning and object 
    recognition algorithms with minimal requirement on data 
    preprocessing and formatting. It can be seen as similar in flavor 
    to MNIST (e.g., the images are of small cropped digits), but 
    incorporates an order of magnitude more labeled data (over 600,000 
    digit images) and comes from a significantly harder, unsolved, 
    real world problem (recognizing digits and numbers in natural 
    scene images). SVHN is obtained from house numbers in Google 
    Street View images. 
    """

    def download(path):
        """
        Download the SVHN dataset and store the result into the given
        path

        Parameters
        ----------

            path: str
                the path where the downloaded files will be stored. If the
                directory does not exist, it is created.
        """

        # Load the dataset (download if necessary) and set
        # the class attributess.
        print("Loading svhn")

        t0 = time.time()

        if not os.path.isdir(path + "svhn"):
            os.mkdir(path + "svhn")
            print("\tCreating svhn Directory")

        if not os.path.exists(path + "svhn/train_32x32.mat"):
            url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
            urllib.request.urlretrieve(url, path + "svhn/train_32x32.mat")

        if not os.path.exists(path + "svhn/test_32x32.mat"):
            url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
            urllib.request.urlretrieve(url, path + "svhn/test_32x32.mat")

    def load(path=None):
        """
        Parameters
        ----------
            path: str (optional)
                default $DATASET_PATH, the path to look for the data and
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

        svhn.download(path)

        # Load the dataset (download if necessary) and set
        # the class attributess.
        print("Loading svhn")

        t0 = time.time()

        # Train set
        data = sio.loadmat(path + "svhn/train_32x32.mat")
        train_images = data["X"].transpose([3, 2, 0, 1])
        train_labels = np.squeeze(data["y"]) - 1

        # Test set
        data = sio.loadmat(path + "svhn/test_32x32.mat")
        test_images = data["X"].transpose([3, 2, 0, 1])
        test_labels = np.squeeze(data["y"]) - 1

        print("Dataset svhn loaded in", "{0:.2f}".format(time.time() - t0), "s.")
        return train_images, train_labels, test_images, test_labels
