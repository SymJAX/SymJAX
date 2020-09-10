import io
import os
import time
from .utils import download_dataset
import zipfile

import numpy as np
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm


_dataset = "audiomnist"
_urls = {"https://github.com/soerenab/AudioMNIST/archive/master.zip": "data.zip"}


def load(path=None):
    """
    digit recognition
        https://github.com/soerenab/AudioMNIST

    A simple audio/speech dataset consisting of recordings of spoken digits in
    wav files at 8kHz. The recordings are trimmed so that they have near
    minimal silence at the beginnings and ends.

    FSDD is an open dataset, which means it will grow over time as data is
    contributed. In order to enable reproducibility and accurate citation the
    dataset is versioned using Zenodo DOI as well as git tags.

    Current status

        4 speakers
        2,000 recordings (50 of each digit per speaker)
        English pronunciations
    """
    if path is None:
        path = os.environ["DATASET_PATH"]
    download_dataset(path, _dataset, _urls)

    t0 = time.time()

    # load wavs
    f = zipfile.ZipFile(os.path.join(path, _dataset, "data.zip"))
    wavs = list()
    digits = list()
    speakers = list()
    N = 0
    for filename in tqdm(f.namelist(), ascii=True):
        if ".wav" not in filename:
            continue
        filename_end = filename.split("/")[-1]
        digits.append(int(filename_end.split("_")[0]))
        speakers.append(int(filename_end.split("_")[1]) - 1)
        wavfile = f.read(filename)
        byt = io.BytesIO(wavfile)
        wavs.append(wav_read(byt)[1].astype("float32"))
        N = max(N, len(wavs[-1]))

    digits = np.array(digits)
    speakers = np.array(speakers)
    all_wavs = np.zeros((len(wavs), N))
    for i in range(len(wavs)):
        left = (N - len(wavs[i])) // 2
        all_wavs[i, left : left + len(wavs[i])] = wavs[i]
    print("Audio-MNIST loaded in {} s.".format(time.time() - t0))
    return all_wavs, digits, speakers
