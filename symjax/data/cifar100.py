import os
import pickle
import tarfile
import time
import urllib.request

import numpy as np


class cifar100:
    """Image classification.

    The `CIFAR-100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ dataset is 
    just like the CIFAR-10, except it has 100 classes containing 600 images 
    each. There are 500 training images and 100 testing images per class. 
    The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each 
    image comes with a "fine" label (the class to which it belongs) and a 
    "coarse" label (the superclass to which it belongs).
    """

    labels_list = [
        "apple",
        "aquarium_fish",
        "baby",
        "bear",
        "beaver",
        "bed",
        "bee",
        "beetle",
        "bicycle",
        "bottle",
        "bowl",
        "boy",
        "bridge",
        "bus",
        "butterfly",
        "camel",
        "can",
        "castle",
        "caterpillar",
        "cattle",
        "chair",
        "chimpanzee",
        "clock",
        "cloud",
        "cockroach",
        "couch",
        "crab",
        "crocodile",
        "cup",
        "dinosaur",
        "dolphin",
        "elephant",
        "flatfish",
        "forest",
        "fox",
        "girl",
        "hamster",
        "house",
        "kangaroo",
        "keyboard",
        "lamp",
        "lawn_mower",
        "leopard",
        "lion",
        "lizard",
        "lobster",
        "man",
        "maple_tree",
        "motorcycle",
        "mountain",
        "mouse",
        "mushroom",
        "oak_tree",
        "orange",
        "orchid",
        "otter",
        "palm_tree",
        "pear",
        "pickup_truck",
        "pine_tree",
        "plain",
        "plate",
        "poppy",
        "porcupine",
        "possum",
        "rabbit",
        "raccoon",
        "ray",
        "road",
        "rocket",
        "rose",
        "sea",
        "seal",
        "shark",
        "shrew",
        "skunk",
        "skyscraper",
        "snail",
        "snake",
        "spider",
        "squirrel",
        "streetcar",
        "sunflower",
        "sweet_pepper",
        "table",
        "tank",
        "telephone",
        "television",
        "tiger",
        "tractor",
        "train",
        "trout",
        "tulip",
        "turtle",
        "wardrobe",
        "whale",
        "willow_tree",
        "wolf",
        "woman",
        "worm",
    ]

    def download(path):
        """
        Download the CIFAR100 dataset and store the result into the given
        path

        Parameters
        ----------

            path: str
                the path where the downloaded files will be stored. If the
                directory does not exist, it is created.
        """

        print("Loading cifar100")
        t0 = time.time()

        if not os.path.isdir(path + "cifar100"):
            print("\tCreating cifar100 Directory")
            os.mkdir(path + "cifar100")

        if not os.path.exists(path + "cifar100/cifar100.tar.gz"):
            url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            with DownloadProgressBar(
                unit="B", unit_scale=True, miniters=1, desc="Downloading dataset"
            ) as t:
                urllib.request.urlretrieve(url, path + "cifar100/cifar100.tar.gz")

    def load(path=None):

        if path is None:
            path = os.environ["DATASET_PATH"]
        cifar100.download(path)

        # Loading the file
        tar = tarfile.open(path + "cifar100/cifar100.tar.gz", "r:gz")

        # Loading training set
        f = tar.extractfile("cifar-100-python/train").read()
        data = pickle.loads(f, encoding="latin1")
        train_images = data["data"].reshape((-1, 3, 32, 32))
        train_fine = np.array(data["fine_labels"])
        train_coarse = np.array(data["coarse_labels"])

        # Loading test set
        f = tar.extractfile("cifar-100-python/test").read()
        data = pickle.loads(f, encoding="latin1")
        test_images = data["data"].reshape((-1, 3, 32, 32))
        test_fine = np.array(data["fine_labels"])
        test_coarse = np.array(data["coarse_labels"])

        print("Dataset cifar100 loaded in {0:.2f}s.".format(time.time() - t0))
        return (
            train_images,
            train_fine,
            train_coarse,
            test_images,
            test_fine,
            test_coarse,
        )
