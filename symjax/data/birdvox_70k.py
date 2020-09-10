#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"
import os
import pickle, gzip
import urllib.request
import numpy as np
import time
import h5py
from tqdm import tqdm
from .utils import download_dataset


_urls = {
    "https://zenodo.org/record/1226427/files/BirdVox-70k_unit{}.hdf5?download=1".format(
        i
    ): "BirdVox-70k_unit{}.hdf5".format(i)
    for i in ["01", "02", "03", "05", "07", "10"]
}


def load(path=None):
    """a dataset for avian flight call detection in half-second clips

    Version 1.0, April 2018.

    Created By

    Vincent Lostanlen (1, 2, 3), Justin Salamon (2, 3), Andrew Farnsworth (1),
    Steve Kelling (1), and Juan Pablo Bello (2, 3).

    (1): Cornell Lab of Ornithology (CLO)
    (2): Center for Urban Science and Progress, New York University
    (3): Music and Audio Research Lab, New York University

    https://wp.nyu.edu/birdvox

    Description

    The BirdVox-70k dataset contains 70k half-second clips from 6 audio
    recordings in the BirdVox-full-night dataset, each about ten hours in
    duration. These recordings come from ROBIN autonomous recording units,
    placed near Ithaca, NY, USA during the fall 2015. They were captured on the
    night of September 23rd, 2015, by six different sensors, originally
    numbered 1, 2, 3, 5, 7, and 10.

    Andrew Farnsworth used the Raven software to pinpoint every avian flight
    call in time and frequency. He found 35402 flight calls in total.
    He estimates that about 25 different species of passerines (thrushes,
    warblers, and sparrows) are present in this recording. Species are not
    labeled in BirdVox-70k, but it is possible to tell apart thrushes from
    warblers and sparrows by looking at the center frequencies of their calls.
    The annotation process took 102 hours.

    The dataset can be used, among other things, for the research,development
    and testing of bioacoustic classification models, including the
    reproduction of the results reported in [1].

    For details on the hardware of ROBIN recording units, we refer the reader
    to [2].

    [1] V. Lostanlen, J. Salamon, A. Farnsworth, S. Kelling, J. Bello. BirdVox-full-night: a dataset and benchmark for avian flight call detection. Proc. IEEE ICASSP, 2018.

    [2] J. Salamon, J. P. Bello, A. Farnsworth, M. Robbins, S. Keen, H. Klinck, and S. Kelling. Towards the Automatic Classification of Avian Flight Calls for Bioacoustic Monitoring. PLoS One, 2016.

    @inproceedings{lostanlen2018icassp,
    title = {BirdVox-full-night: a dataset and benchmark for avian flight call detection},
    author = {Lostanlen, Vincent and Salamon, Justin and Farnsworth, Andrew and Kelling, Steve and Bello, Juan Pablo},
    booktitle = {Proc. IEEE ICASSP},
    year = {2018},
    published = {IEEE},
    venue = {Calgary, Canada},
    month = {April},
    }

    Parameters
    ----------
        path: str (optional)
            default ($DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present

    Returns
    -------

        wavs: array(70804, 12000)
            the waveforms in the time amplitude domain

        labels: array(70804,)
            binary values representing the presence or not of an avian

        recording: array(70804,)
            the file number from which the sample has been extracted

    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    if path is None:
        path = os.environ["DATASET_PATH"]
    download_dataset(path, "birdvox_70k", _urls)

    # Loading the file
    path += "birdvox_70k/"
    names = ["01", "02", "03", "05", "07", "10"]
    basefile = "BirdVox-70k_unit{}.hdf5"
    wavs = list()
    label = list()
    recording = list()
    for name in names:
        f = h5py.File(path + basefile.format(name), "r")
        for filename in tqdm(
            f["waveforms"].keys(), ascii=True, desc="recording {}".format(name)
        ):
            wavs.append(f["waveforms"][filename][...])
            label.append(int(filename[-1]))
            recording.append(int(name))

    data = {
        "wavs": np.array(wavs).astype("float32"),
        "labels": np.array(label).astype("int32"),
        "recording": np.array(recording).astype("int32"),
    }

    print("Dataset birdvox_70k loaded in {0:.2f}s.".format(time.time() - t0))

    return data
