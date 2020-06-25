#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io
import os
import time
import zipfile
import matplotlib.image as mpimg
import urllib
import numpy as np


__author__ = "Randall Balestriero"


class cassava:
    """Plant images classification.

    The data consists of two folders, a training folder that contains 5
    subfolders that contain the respective images for the different 5 classes
    and a test folder containing test images.

    Participants are to train their models using the images in the training
    folder and provide a submission file like the sample provided which contains
    the image name exactly matching the image name in the test folder and the
    corresponding class prediction with labels corresponding to the disease
    categories, cmd, healthy, cgm, cbsd, cbb.

    Please cite this paper if you use the dataset for your project:
    https://arxiv.org/pdf/1908.02900.pdf

    """

    classes = ["cbb", "cmd", "cbsd", "cgm", "healthy"]

    @staticmethod
    def download(path):
        """
        Download the cassava dataset and store the result into the given
        path

        Parameters
        ----------

            path: str
                the path where the downloaded files will be stored. If the
                directory does not exist, it is created.
        """

        # Check if directory exists
        if not os.path.isdir(path + "cassava"):
            print("Creating cassava Directory")
            os.mkdir(path + "cassava")
        # Check if file exists
        if not os.path.exists(path + "cassava/cassavaleafdata.zip"):
            url = (
                "https://storage.googleapis.com/emcassavadata/" + "cassavaleafdata.zip"
            )
            urllib.request.urlretrieve(url, path + "cassava/cassavaleafdata.zip")

    @staticmethod
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

            valid_images: array

            valid_labels: array

            test_images: array

            test_labels: array

        """

        if path is None:
            path = os.environ["DATASET_PATH"]

        cassava.download(path)

        t0 = time.time()

        # Loading the file
        data = {"train": [[], []], "test": [[], []], "validation": [[], []]}

        f = zipfile.ZipFile(path + "cassava/cassavaleafdata.zip")
        for filename in f.namelist():
            if ".jpg" not in filename:
                continue
            setname, foldername = filename.split("/")[1:3]
            img = mpimg.imread(io.BytesIO(f.read(filename)), "jpg")
            data[setname][0].append(img)
            data[setname][1].append(cassava.classes.index(foldername))

        train_images = np.array(data["train"][0])
        test_images = np.array(data["test"][0])
        valid_images = np.array(data["validation"][0])

        train_labels = np.array(data["train"][1])
        test_labels = np.array(data["test"][1])
        valid_labels = np.array(data["validation"][1])

        print("Dataset cassava loaded in {0:.2f}s.".format(time.time() - t0))

        return (
            train_images,
            train_labels,
            valid_images,
            valid_labels,
            test_images,
            test_labels,
        )
