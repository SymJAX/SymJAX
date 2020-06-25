import os
from tqdm import tqdm
import urllib.request
import numpy as np
import time
import io
import imageio
import tarfile
from scipy.ndimage import imread


class cub200:
    """ Image classification of bird species.
    The `CUB-200 <http://www.vision.caltech.edu/visipedia/CUB-200-2011.html>`_.
    dataset  contains  11,788  images of  200  bird  species
    This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds 
    from 10 classes: air_conditioner, car_horn, children_playing, 
    dog_bark, drilling, enginge_idling, gun_shot, jackhammer, siren, 
    and street_music. The classes are drawn from the 
    `urban sound taxonomy <https://urbansounddataset.weebly.com/taxonomy.html>`_. 
    The dataset is obtained from `Kaggle <https://www.kaggle.com/pavansanagapati/urban-sound-classification>_`
    """

    def __init__(self, target="class", data_format="NCHW"):
        """set up the options
        target = {class,bounding_box}
        crop-images True or False (whether to use the cropped images)
        """

    data_format = "NCHW"
    """(default)"""
    datum_shape = (3, None, None)
    """(default)"""
    given_test_set = True
    """"""
    given_valid_set = False
    """"""
    given_unlabeled = False
    """"""
    n_classes = 200
    """"""
    # To be defined during loading
    classes = {}
    name = "cub200"

    def load(data_format="NCHW"):
        """Load the dataset (download if necessary) and adapt
        the class attributes based on the given data format.

        :param data_format: (optional, default 'NCHW'), if different than default, adapts :mod:`data_format` and :mod:`datum_shape`
        :type data_format: 'NCHW' or 'NHWC'
        :return: return the train and test set, each as a couple (images,labels)
        :rtype: [(train_images,train_labels),
                (test_images,test_labels)]
        """
        t = time.time()

        PATH = os.environ["DATASET_PATH"]

        if not os.path.isdir(PATH + "cub200"):
            print("Creating Directory")
            os.mkdir(PATH + "cub200")

        if not os.path.exists(PATH + "cub200/CUB_200_2011.tgz"):
            td = time.time()
            print("\tDownloading Data")
            url = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
            urllib.request.urlretrieve(url, PATH + "cub200/CUB_200_2011.tgz")
            print("\tDone in {:.2f}".format(time.time() - td))

        print("Loading caltechbird")
        tar = tarfile.open(PATH + "cub200/CUB_200_2011.tgz", "r:gz")

        # Load the class names
        f = tar.extractfile("CUB_200_2011/classes.txt")
        names = np.loadtxt(f, dtype="str")
        cub200.classes = dict([[c, n.split(".")[1]] for c, n in enumerate(names[:, 1])])

        # Load Bounding boxes
        f = tar.extractfile("CUB_200_2011/bounding_boxes.txt")
        boxes = np.loadtxt(f, dtype="int32")
        bounding_boxes = dict()
        for i in range(boxes.shape[0]):
            bounding_boxes[str(boxes[i, 0])] = boxes[i, 1:]

        # Load dataset
        labels = list()
        boxes = list()
        data = np.empty((11788, 3, 500, 500)) * np.nan
        cpt = 0
        nb_removed_images = 0
        for member in tqdm(tar.getmembers()):
            if "CUB_200_2011/images/" in member.name and "jpg" in member.name:
                class_ = member.name.split("/")[2].split(".")[0]
                image_id = member.name.split("_")[-1][:-4]
                f = tar.extractfile(member)
                im = imageio.imread(f, format="jpg")
                if len(im.shape) == 2:
                    nb_removed_images += 1
                    print("i")
                    continue
                data[cpt, :, : im.shape[0], : im.shape[1]] = im.transpose([2, 0, 1])
                labels.append(int(class_))
                #                boxes.append(bounding_boxes[image_id])
                cpt += 1
        if nb_removed_images:
            data = data[:-nb_removed_images]
        labels = np.array(labels).astype("int32")
        if data_format == "NHWC":
            cub200.datum_shape = (None, None, 3)
            cub200.data_format = data_format
            data = data.transpose([0, 2, 3, 1])

        print("Dataset cub200 loaded in", "{0:.2f}".format(time.time() - t), "s.")

        return data, labels, np.asarray(boxes)
