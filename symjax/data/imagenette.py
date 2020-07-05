import os
import tarfile
import time
import urllib.request
import matplotlib.image
from multiprocessing import Pool
from functools import partial

import numpy as np
from tqdm import tqdm


__DESCRIPTION__ = """Imagenette is a subset of 10 easily classified classes 
from Imagenet (tench, English springer, cassette player, chain saw, church, 
French horn, garbage truck, gas pump, golf ball, parachute).

'Imagenette' is pronounced just like 'Imagenet', except with a corny 
inauthentic French accent. If you've seen Peter Sellars in The Pink Panther,
 then think something like that. It's important to ham up the accent as much as
  possible, otherwise people might not be sure whether you're refering to 
  "Imagenette" or "Imagenet". (Note to native French speakers: to avoid
   confusion, be sure to use a corny inauthentic American accent when saying 
   "Imagenet". Think something like the philosophy restaurant skit from Monty
    Python's The Meaning of Life.)
"""


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
    if not os.path.isdir(path + "imagenette"):
        print("\tCreating imagenette Directory")
        os.mkdir(path + "imagenette")

    # Check if file exists
    if not os.path.exists(path + "imagenette/imagenette2.tgz"):
        print("\tDownloading imagenette")
        url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
        urllib.request.urlretrieve(url, path + "imagenette/imagenette2.tgz")


def load_image(tar, name):
    file = tar.extractfile(name)
    image = matplotlib.image.imread(file, "JPEG")
    print("image", image)
    return image


def load(path=None, n_processes=6):
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

    download(path)

    t0 = time.time()

    print("Extracting...")
    tar = tarfile.open(path + "imagenette/imagenette2.tgz", "r:gz")
    tar.extractall(path + "imagenette/")

    # Load train set
    train_images = list()
    train_labels = list()
    test_images = list()
    test_labels = list()

    train_names = [
        name for name in tar.getnames() if "JPEG" in name and "train" in name
    ]
    test_names = [
        name for name in tar.getnames() if "JPEG" in name and "train" not in name
    ]

    for name in tqdm(train_names, ascii=True, desc="Loading training set"):
        train_images.append(matplotlib.image.imread(path + "imagenette/" + name))
        train_labels.append(name.split("/")[2])

    for name in tqdm(test_names, ascii=True, desc="Loading test set"):
        test_images.append(matplotlib.image.imread(path + "imagenette/" + name))
        test_labels.append(name.split("/")[2])

    data = {
        "train_set/images": train_images,
        "train_set/labels": train_labels,
        "test_set/images": test_images,
        "test_set/labels": test_labels,
    }

    tar.close()

    print("Dataset imagenette loaded in{0:.2f}s.".format(time.time() - t0))

    return data
