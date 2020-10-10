import os
from .utils import download_dataset
import time
import zipfile
import imageio
from tqdm import tqdm
import numpy as np


_dataset = "rps"

_urls = {
    "https://storage.googleapis.com/download.tensorflow.org/data/rps.zip": "rps.zip",
    "https://storage.googleapis.com/download.tensorflow.org/data/rps-test-set.zip": "rps-test-set.zip",
}


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
    test_images = []
    test_classes = []
    test_styles = []
    train_images = []
    train_classes = []
    train_styles = []
    with zipfile.ZipFile(
        os.path.join(path, _dataset, "rps-test-set.zip"), "r"
    ) as zfile:
        for filename in tqdm(zfile.namelist(), desc="test set", ascii=True):
            if ".png" not in filename:
                continue
            test_classes.append(filename.split("/")[1])
            test_styles.append(filename.split("-")[-2][-2:])
            test_images.append(imageio.imread(zfile.read(filename)))

    with zipfile.ZipFile(os.path.join(path, _dataset, "rps.zip"), "r") as zfile:
        for filename in tqdm(zfile.namelist(), desc="train set", ascii=True):
            if ".png" not in filename:
                continue
            train_classes.append(filename.split("/")[1])
            train_styles.append(filename.split("-")[0][-2:])
            train_images.append(imageio.imread(zfile.read(filename)))

    data = {
        "train_set/images": np.array(train_images),
        "train_set/labels": np.array(train_classes),
        "train_set/styles": np.array(train_styles),
        "test_set/images": np.array(test_images),
        "test_set/labels": np.array(test_classes),
        "test_set/styles": np.array(test_styles),
    }

    print("Dataset rps loaded in {0:.2f}s.".format(time.time() - t0))

    return data
