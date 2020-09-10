import os
import tarfile
import time
import matplotlib.image
from .utils import download_dataset

import numpy as np
from tqdm import tqdm


imagenette_map = {
    "n01440764": "tench",
    "n02102040": "springer",
    "n02979186": "casette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "French_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}

_dataset = "imagenette"
_urls = {
    "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz": "imagenette2.tgz"
}


def load(path=None, n_processes=6):
    """10-class image classification form imagenet
    Imagenette is a subset of 10 easily classified classes
    from Imagenet (tench, English springer, cassette player, chain saw, church,
    French horn, garbage truck, gas pump, golf ball, parachute).

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

    download_dataset(path, _dataset, _urls, extract=True)

    t0 = time.time()

    # Load train set
    train_images = list()
    train_labels = list()
    test_images = list()
    test_labels = list()

    tar = tarfile.open(path + "imagenette/imagenette2.tgz", "r:gz")

    train_names = [
        name for name in tar.getnames() if "JPEG" in name and "train" in name
    ]
    test_names = [
        name for name in tar.getnames() if "JPEG" in name and "train" not in name
    ]

    for name in tqdm(train_names, ascii=True, desc="Loading training set"):
        image = matplotlib.image.imread(path + "imagenette/" + name)
        if image.ndim == 2:
            image = image[:, :, None] * np.ones((1, 1, 3))
        train_images.append(image)
        train_labels.append(name.split("/")[2])

    for name in tqdm(test_names, ascii=True, desc="Loading test set"):
        image = matplotlib.image.imread(path + "imagenette/" + name)
        if image.ndim == 2:
            image = image[:, :, None] * np.ones((1, 1, 3))
        test_images.append(image)
        test_labels.append(name.split("/")[2])

    train_labels = (np.unique(train_labels) == np.array(train_labels)[:, None]).argmax(
        1
    )
    test_labels = (np.unique(test_labels) == np.array(test_labels)[:, None]).argmax(1)

    data = {
        "train_set/images": train_images,
        "train_set/labels": train_labels.astype("int32"),
        "test_set/images": test_images,
        "test_set/labels": test_labels.astype("int32"),
    }

    tar.close()

    print("Dataset imagenette loaded in{0:.2f}s.".format(time.time() - t0))

    return data
