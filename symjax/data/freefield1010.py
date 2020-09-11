import os
import numpy as np
import zipfile
import io
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm
from .utils import download_dataset


_urls = {
    "hhttps://archive.org/download/ff1010bird/ff1010bird_wav.zip": "ff1010bird_wav.zip",
    "https://ndownloader.figshare.com/files/6035814": "ff1010bird_metadata.csv",
}


def load(path=None):
    """Audio binary classification, presence or absence of bird songs.
    `freefield1010 <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_.
    is a collection of over 7,000 excerpts from field recordings
    around the world, gathered by the FreeSound project, and then standardised
    for research. This collection is very diverse in location and environment,
    and for the BAD Challenge we have newly annotated it for the
    presence/absence of birds.
    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, "freefield1010", _urls)

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
    wavs = np.empty((N, 441000), dtype="float32")
    for i, files_ in tqdm(enumerate(labels[:, 0]), ascii=True, total=N):
        wavfile = f.read("wav/" + str(files_) + ".wav")
        byt = io.BytesIO(wavfile)
        wavs[i] = wav_read(byt)[1].astype("float32")

    labels = labels[:, 1]

    data = {"wavs": wavs, "labels": labels}
    return data
