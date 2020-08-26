import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm


class warblr:
    """Binary audio classification, presence or absence of a bird.

    `Warblr <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_
    comes from a UK bird-sound crowdsourcing
    research spinout called Warblr. From this initiative we have
    10,000 ten-second smartphone audio recordings from around the UK.
    The audio totals around 44 hours duration. The audio will be
    published by Warblr under a Creative Commons licence. The audio
    covers a wide distribution of UK locations and environments, and
    includes weather noise, traffic noise, human speech and even human
    bird imitations. It is directly representative of the data that is
    collected from a mobile crowdsourcing initiative.
    """

    def download(path):
        """
        Download the data
        """

        # Load the dataset (download if necessary) and set
        # the class attributes.
        print("Loading warblr")
        t = time.time()
        if not os.path.isdir(path + "warblr"):
            print("\tCreating Directory")
            os.mkdir(path + "warblr")

        if not os.path.exists(path + "warblr/warblrb10k_public_wav.zip"):
            url = "https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip"
            urllib.request.urlretrieve(url, path + "warblr/warblrb10k_public_wav.zip")

        if not os.path.exists(path + "warblr/warblrb10k_public_metadata.csv"):
            url = "https://ndownloader.figshare.com/files/6035817"
            urllib.request.urlretrieve(
                url, path + "warblr/warblrb10k_public_metadata.csv"
            )

    def load(path=None):
        """
        Load the data given a path
        """

        if path is None:
            path = os.environ["DATASET_PATH"]

        warblr.download(path)

        # Load the dataset (download if necessary) and set
        # the class attributes.
        print("Loading warblr")
        t = time.time()

        # Loading Labels
        labels = np.loadtxt(
            path + "warblr/warblrb10k_public_metadata.csv",
            delimiter=",",
            skiprows=1,
            dtype="str",
        )
        # Loading the files
        f = zipfile.ZipFile(path + "warblr/warblrb10k_public_wav.zip")
        N = labels.shape[0]
        wavs = list()
        for i, files_ in tqdm(enumerate(labels), ascii=True):
            wavfile = f.read("wav/" + files_[0] + ".wav")
            byt = io.BytesIO(wavfile)
            wavs.append(np.expand_dims(wav_read(byt)[1].astype("float32"), 0))
        labels = labels[:, 1].astype("int32")

        print("Dataset warblr loaded in", "{0:.2f}".format(time.time() - t), "s.")
        return wavs, labels
