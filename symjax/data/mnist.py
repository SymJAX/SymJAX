import os
import pickle
import gzip
from .utils import download_dataset
import time

_dataset = "mnist"

_urls = {"http://deeplearning.net/data/mnist/mnist.pkl.gz": "mnist.pkl.gz"}


def load(path=None):
    """

    The MNIST database of handwritten digits, available from this page
    has a training set of 60,000 examples, and a test set of 10,000 examples.
    It is a subset of a larger set available from NIST. The digits have been
    size-normalized and centered in a fixed-size image.

    It is a good database for people who want to try learning techniques
    and pattern recognition methods on real-world data while spending minimal
    efforts on preprocessing and formatting.

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

    download_dataset(path, _dataset, _urls)

    t0 = time.time()

    # Loading the file
    print("Loading mnist")
    f = gzip.open(os.path.join(path, "mnist/mnist.pkl.gz"), "rb")
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
