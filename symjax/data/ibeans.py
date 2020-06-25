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


class ibeans:
    """Plant images classification.

    This dataset is of leaf images taken in the field in different
    districts in Uganda by the Makerere AI lab in collaboration with the
    National Crops Resources Research Institute (NaCRRI), the national
    body in charge of research in agriculture in Uganda.

    The goal is to build a robust machine learning model that is able
    to distinguish between diseases in the Bean plants. Beans are an
    important cereal food crop for Africa grown by many small-holder
    farmers - they are a significant source of proteins for school-age
    going children in East Africa.

    The data is of leaf images representing 3 classes: the healthy class of
    images, and two disease classes including Angular Leaf Spot and Bean
    Rust diseases. The model should be able to distinguish between these 3
    classes with high accuracy. The end goal is to build a robust, model
    that can be deployed on a mobile device and used in the field by a
    farmer.

    The data includes leaf images taken in the field. The figure above
    depicts examples of the types of images per class. Images were taken
    from the field/garden a basic smartphone.

    The images were then annotated by experts from NaCRRI who determined
    for each image which disease was manifested. The experts were part of
    the data collection team and images were annotated directly during the
    data collection process in the field.

    Class   Examples
    Healthy class   428
    Angular Leaf Spot   432
    Bean Rust   436
    Total:  1,296

    Data Released   20-January-2020
    License     MIT
    Credits     Makerere AI Lab
    """

    classes = ["angular_leaf_spot", "bean_rust", "healthy"]

    @staticmethod
    def download(path):
        """
        Download the ibeans dataset and store the result into the given
        path

        Parameters
        ----------

            path: str
                the path where the downloaded files will be stored. If the
                directory does not exist, it is created.
        """

        # Check if directory exists
        if not os.path.isdir(path + "ibeans"):
            print("Creating mnist Directory")
            os.mkdir(path + "ibeans")
        # Check if file exists
        if not os.path.exists(path + "ibeans/train.zip"):
            url = "https://storage.googleapis.com/ibeans/train.zip"
            urllib.request.urlretrieve(url, path + "ibeans/train.zip")

        # Check if file exists
        if not os.path.exists(path + "ibeans/test.zip"):
            url = "https://storage.googleapis.com/ibeans/test.zip"
            urllib.request.urlretrieve(url, path + "ibeans/test.zip")

        # Check if file exists
        if not os.path.exists(path + "ibeans/validation.zip"):
            url = "https://storage.googleapis.com/ibeans/validation.zip"
            urllib.request.urlretrieve(url, path + "ibeans/validation.zip")

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

        ibeans.download(path)

        t0 = time.time()

        # Loading the file
        train_images = list()
        train_labels = list()
        f = zipfile.ZipFile(path + "ibeans/train.zip")
        for filename in f.namelist():
            if ".jpg" not in filename:
                continue
            train_images.append(mpimg.imread(io.BytesIO(f.read(filename)), "jpg"))
            train_labels.append(ibeans.classes.index(filename.split("/")[1]))

        # Loading the file
        test_images = list()
        test_labels = list()
        f = zipfile.ZipFile(path + "ibeans/test.zip")
        for filename in f.namelist():
            if ".jpg" not in filename:
                continue
            test_images.append(mpimg.imread(io.BytesIO(f.read(filename)), "jpg"))
            test_labels.append(ibeans.classes.index(filename.split("/")[1]))

        # Loading the file
        valid_images = list()
        valid_labels = list()
        f = zipfile.ZipFile(path + "ibeans/validation.zip")
        for filename in f.namelist():
            if ".jpg" not in filename:
                continue
            valid_images.append(mpimg.imread(io.BytesIO(f.read(filename)), "jpg"))
            valid_labels.append(ibeans.classes.index(filename.split("/")[1]))

        train_images = np.array(train_images)
        test_images = np.array(test_images)
        valid_images = np.array(valid_images)

        train_labels = np.array(train_labels)
        test_labels = np.array(test_labels)
        valid_labels = np.array(valid_labels)

        print("Dataset ibeans loaded in {0:.2f}s.".format(time.time() - t0))

        return (
            train_images,
            train_labels,
            valid_images,
            valid_labels,
            test_images,
            test_labels,
        )
