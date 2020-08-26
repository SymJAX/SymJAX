import os
import pickle, gzip
import urllib.request
import numpy as np
import time
import tarfile
from tqdm import tqdm
import zipfile
from scipy.io.wavfile import read as wav_read
import io


class FSDKaggle2018:
    """FSDKaggle2018 Sound Classification
    https://zenodo.org/record/2552860
    """

    def download(path):

        # Check if directory exists
        if not os.path.isdir(path + "FSDKaggle2018"):
            print("\tCreating FSDKaggle2018 Directory")
            os.mkdir(path + "FSDKaggle2018")

        url = "https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_test.zip?download=1"
        # Check if file exists
        if not os.path.exists(path + "FSDKaggle2018/audio_test.zip"):
            print("\tDownloading test set")
            urllib.request.urlretrieve(url, path + "FSDKaggle2018/audio_test.zip")
        url = "https://zenodo.org/record/2552860/files/FSDKaggle2018.audio_train.zip?download=1"
        if not os.path.exists(path + "FSDKaggle2018/audio_train.zip"):
            print("\tDownloading train set")
            urllib.request.urlretrieve(url, path + "FSDKaggle2018/audio_train.zip")
        url = (
            "https://zenodo.org/record/2552860/files/FSDKaggle2018.meta.zip?download=1"
        )
        if not os.path.exists(path + "FSDKaggle2018/meta.zip"):
            print("\tDownloading meta set")
            urllib.request.urlretrieve(url, path + "FSDKaggle2018/meta.zip")

    def load(path=None):
        """
        Parameters
        ----------

        path: str (optional)
                default $DATASET_path), the path to look for the data and
                where the data will be downloaded if not present

        Returns
        -------

        wavs: array
            the wavs as a numpy array (matrix) with first dimension the data
            and second dimension time

        labels: array
            the labels of the final classes (41 different ones) as a integer
            vector
        """

        if path is None:
            path = os.environ["DATASET_PATH"]
        FSDKaggle2018.download(path)
        t0 = time.time()

        f = zipfile.ZipFile(path + "FSDKaggle2018/audio_train.zip")
        wavs_train = list()
        names_train = list()
        for filename in tqdm(f.namelist(), ascii=True, desc="Loading train set"):
            if ".wav" not in filename:
                continue
            wavfile = f.read(filename)
            byt = io.BytesIO(wavfile)
            wavs_train.append(wav_read(byt)[1].astype("float32"))
            names_train.append((filename.split("/")[-1]))

        f = zipfile.ZipFile(path + "FSDKaggle2018/audio_test.zip")
        wavs_test = list()
        names_test = list()
        for filename in tqdm(f.namelist(), ascii=True, desc="Loading test set"):
            if ".wav" not in filename:
                continue
            wavfile = f.read(filename)
            byt = io.BytesIO(wavfile)
            wavs_test.append(wav_read(byt)[1].astype("float32"))
            names_test.append((filename.split("/")[-1]))

        f = zipfile.ZipFile(path + "FSDKaggle2018/meta.zip")
        meta_train = np.loadtxt(
            io.BytesIO(f.read("FSDKaggle2018.meta/train_post_competition.csv")),
            delimiter=",",
            skiprows=1,
            dtype="str",
        )
        meta_test = np.loadtxt(
            io.BytesIO(
                f.read("FSDKaggle2018.meta/test_post_competition_scoring_clips.csv")
            ),
            delimiter=",",
            skiprows=1,
            dtype="str",
        )

        filenames = list(meta_train[:, 0])
        labels_train, verified, fsid_train = [], [], []
        for i in range(len(wavs_train)):
            index = filenames.index(names_train[i])
            labels_train.append(meta_train[index][1])
            verified.append(meta_train[index][2])
            fsid_train.append(meta_train[index][3])

        filenames = list(meta_test[:, 0])
        labels_test, usage, fsid_test = [], [], []
        for i in range(len(wavs_test)):
            index = filenames.index(names_test[i])
            labels_test.append(meta_test[index][1])
            usage.append(meta_test[index][2])
            fsid_test.append(meta_test[index][3])
        dataset = {
            "wavs_train": wavs_train,
            "labels_train": labels_train,
            "verified_train": verified,
            "fsid_train": fsid_train,
            "wavs_test": wavs_test,
            "labels_test": labels_test,
            "usage_test": usage,
            "fsid_test": fsid_test,
        }
        return dataset
