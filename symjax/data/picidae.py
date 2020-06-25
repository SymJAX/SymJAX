#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"
import os
import io
import urllib.request
import numpy as np
import time
import h5py
from tqdm import tqdm
import zipfile
from scipy.io.wavfile import read as wav_read


class picidae:
    """An Annotated Acoustic Dataset of 7 Picidae Species

    The proposed dataset contains 1669 labeled audio files from
    the following Picidae species doing 3 types of birdsongs:
    call, drumming and song. Audio data are organized in thirteen
    folders: twelve containing the labeled audios of each
    birdsong type from every Picidae specie listed in what
    follows and one folder containing the background sound samples

    1- DendrocoposLeucotos - call.

    2- DendrocoposLeucotos - drumming.

    3- DendrocoposMajor - call.

    4- DendrocoposMajor - drumming.

    5- DendrocoposMedius - call.

    6- DendrocoposMedius - song.

    7- DendrocoposMinor - call.

    8- DendrocoposMinor - drumming.

    9- DryocopusMartius - call.

    10- DryocopusMartius - drumming.

    11- JynxTorquilla - song.

    12- PicusViridis - song.

    13- Silence (or background noise).

   """

    @staticmethod
    def download(path):
        """
        Download the dataset and store the result into the given
        path

        Parameters
        ----------

            path: str
                the path where the downloaded files will be stored. If the
                directory does not exist, it is created.
        """

        # Check if directory exists
        if not os.path.isdir(path + "picidae"):
            print("Creating picidae Directory")
            os.mkdir(path + "picidae")
        url = "https://zenodo.org/record/574438/files/PicidaeDataset.zip?download=1"
        if not os.path.exists(path + "picidae/PicidaeDataset.zip"):
            urllib.request.urlretrieve(url, path + "picidae/PicidaeDataset.zip")

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

            wavs: array
                the waveforms in the time amplitude domain

            labels: array
                binary values representing the presence or not of an avian

            flag: array
                the Xeno-Canto ID

        """

        if path is None:
            path = os.environ["DATASET_PATH"]

        picidae.download(path)

        t0 = time.time()

        archive = zipfile.ZipFile(path + "picidae/PicidaeDataset.zip")
        wavs = list()
        labels = list()
        XC = list()
        for item in tqdm(archive.namelist(), ascii=True):
            if item[-4:] == ".wav" and "._" not in item:
                wavfile = archive.read(item)
                byt = io.BytesIO(wavfile)
                wavs.append(wav_read(byt)[1].astype("float32"))
                labels.append(item.split("/")[1])
                XC.append(item.split("/")[2].split("-")[0])

        labels = np.array(labels)
        unique = np.unique(labels)
        y = np.zeros(len(labels), dtype="int32")
        for k, name in enumerate(np.sort(unique)):
            y[labels == name] = k

        data = {
            "wavs": wavs,
            "labels": y,
            "names": labels,
            "XC_identifiers": XC,
            "INFOS": picidae.__doc__,
        }

        print("Dataset picidae loaded in {0:.2f}s.".format(time.time() - t0))

        return data
