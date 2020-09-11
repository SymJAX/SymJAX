import os
import zipfile
import io
from tqdm import tqdm
import numpy as np
import time
from scipy.io.wavfile import read as wav_read
from .utils import download_dataset


_urls = {
    "https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1": "IRMAS-TrainingData.zip",
    "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip?download=1": "IRMAS-TestingData-Part1.zip",
    "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip?download=1": "IRMAS-TestingData-Part2.zip",
    "https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip?download=1": "IRMAS-TestingData-Part3.zip",
}


def load(path=None):
    """music instrument classification

    ref https://zenodo.org/record/1290750#.WzCwSRyxXMU

    This dataset includes musical audio excerpts with annotations of the
    predominant instrument(s) present. It was used for the evaluation in the
    following article:

    Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. “A Comparison of Sound
    Segregation Techniques for Predominant Instrument Recognition in Musical
    Audio Signals”, in Proc. ISMIR (pp. 559-564), 2012

    Please Acknowledge IRMAS in Academic Research

    IRMAS is intended to be used for training and testing methods for the
    automatic recognition of predominant instruments in musical audio. The
    instruments considered are: cello, clarinet, flute, acoustic guitar,
    electric guitar, organ, piano, saxophone, trumpet, violin, and human singing
    voice. This dataset is derived from the one compiled by Ferdinand Fuhrmann
    in his PhD thesis, with the difference that we provide audio data in stereo
    format, the annotations in the testing dataset are limited to specific
    pitched instruments, and there is a different amount and lenght of excerpts.
    """
    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, "irmas", _urls)

    t0 = time.time()

    train_wavs = list()
    train_labels = list()
    test_wavs = list()
    test_labels = list()

    # loading the training set
    f = zipfile.ZipFile(path + "irmas/IRMAS-TrainingData.zip")
    namelist = f.namelist()
    for filename in tqdm(namelist, ascii=True):
        if ".wav" not in filename:
            continue
        wavfile = f.read(filename)
        byt = io.BytesIO(wavfile)
        train_wavs.append(wav_read(byt)[1].astype("float32"))
        train_labels.append(filename.split("/")[-2])

    base = "irmas/IRMAS-TestingData-Part{}.zip"
    for part in ["1", "2", "3"]:
        f = zipfile.ZipFile(path + base.format(part))
        namelist = f.namelist()
        for filename in tqdm(namelist, ascii=True, desc="Test data {}/3".format(part)):
            if ".wav" not in filename:
                continue

            byt = io.BytesIO(f.read(filename))
            test_wavs.append(wav_read(byt)[1].astype("float32"))

            byt = io.BytesIO(f.read(filename.replace(".wav", ".txt")))
            test_labels.append(np.loadtxt(byt, dtype="str"))

    data = {
        "train_set/wavs": np.array(train_wavs),
        "train_set/labels": train_labels,
        "test_wavs": np.array(test_wavs),
        "test_labels": test_labels,
    }

    print("Dataset IRMAS loaded in {0:.2f}s.".format(time.time() - t0))
    return data
