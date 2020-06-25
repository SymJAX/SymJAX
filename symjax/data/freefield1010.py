import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm


class freefield1010:
    """Audio binary classification, presence or absence of bird songs.
    `freefield1010 <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_. 
    is a collection of over 7,000 excerpts from field recordings 
    around the world, gathered by the FreeSound project, and then standardised 
    for research. This collection is very diverse in location and environment, 
    and for the BAD Challenge we have newly annotated it for the 
    presence/absence of birds.
    """

    def download(path):

        if path is None:
            path = os.environ["DATASET_path"]

        # Load the dataset (download if necessary) and set
        # the class attributes.

        print("Downloading freefield1010")
        t = time.time()

        if not os.path.isdir(path + "freefield1010"):
            print("\tCreating Directory")
            os.mkdir(path + "freefield1010")

        if not os.path.exists(path + "freefield1010/ff1010bird_wav.zip"):
            url = "https://archive.org/download/ff1010bird/ff1010bird_wav.zip"
            urllib.request.urlretrieve(url, path + "freefield1010/ff1010bird_wav.zip")

        if not os.path.exists(path + "freefield1010/ff1010bird_metadata.csv"):
            url = "https://ndownloader.figshare.com/files/6035814"
            urllib.request.urlretrieve(
                url, path + "freefield1010/ff1010bird_metadata.csv"
            )

        print("freefield1010 downloaded in {} sec.".format(time.time() - t))

    def load(path=None):
        if path is None:
            path = os.environ["DATASET_path"]

        freefield1010.download(path)
        t = time.time()

        # load labels
        labels = np.loadtxt(
            path + "freefield1010/ff1010bird_metadata.csv",
            delimiter=",",
            skiprows=1,
            dtype="int32",
        )
        # load wavs
        f = zipfile.ZipFile(path + "freefield1010/ff1010bird_wav.zip")
        # init. the data array
        N = labels.shape[0]
        wavs = np.empty((N, 441000 // subsample), dtype="float32")
        for i, files_ in tqdm(enumerate(labels[:, 0]), ascii=True, total=N):
            wavfile = f.read("wav/" + str(files_) + ".wav")
            byt = io.BytesIO(wavfile)
            wavs[i] = wav_read(byt)[1].astype("float32")[::subsample]

        labels = labels[:, 1]

        return wavs, labels
