import os
import pickle, gzip
import urllib.request
import numpy as np
import tarfile
import time
from tqdm import tqdm
from scipy.io.wavfile import read as wav_read
from .utils import download_dataset


name2class = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9,
}

_urls = {"http://opihi.cs.uvic.ca/sound/genres.tar.gz": "genres.tar.gz"}


def load(path=None):
    """music genre classification

    This dataset was used for the well known paper in genre classification
    "Musical genre classification of audio signals" by G. Tzanetakis
    and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

    Unfortunately the database was collected gradually and very early on in my
    research so I have no titles (and obviously no copyright permission etc).
    The files were collected in 2000-2001 from a variety of sources including
    personal CDs, radio, microphone recordings, in order to represent a variety
    of recording conditions. Nevetheless I have been providing it to researchers
    upon request mainly for comparison purposes etc. Please contact George
    Tzanetakis (gtzan@cs.uvic.ca) if you intend to publish experimental results
    using this dataset.

    There are some practical and conceptual issues with this dataset, described
    in "The GTZAN dataset: Its contents, its faults, their effects on
    evaluation, and its future use" by B. Sturm on arXiv 2013.
    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, "gtzan", _urls)
    print("Loading gtzan")
    t0 = time.time()
    tar = tarfile.open(path + "gtzan/genres.tar.gz", "r:gz")

    # Load train set
    train_songs = list()
    train_labels = list()
    names = tar.getnames()
    names = tar.getmembers()
    for name in tqdm(names, ascii=True, total=1000):
        if "wav" not in name.name:
            continue
        f = tar.extractfile(name.name)  # .read()
        train_songs.append(wav_read(f)[1])
        t = name.name.split("/")[1]
        train_labels.append(gtzan.name2class[t])

    N = np.min([len(w) for w in train_songs])
    train_songs = [w[:N] for w in train_songs]

    train_songs = np.stack(train_songs).astype("float32")
    train_labels = np.array(train_labels).astype("int32")

    print("Dataset gtzan loaded in{0:.2f}s.".format(time.time() - t0))
    data = {"wavs": train_songs, "labels": train_labels}
    return data
