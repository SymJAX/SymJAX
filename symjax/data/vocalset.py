import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm


class vocalset:
    """singer/technique/vowel of singing voices

    source: https://zenodo.org/record/1442513#.W7OaFBNKjx4

    We present VocalSet, a singing voice dataset consisting of 10.1 hours
    of monophonic recorded audio of professional singers demonstrating both
    standard and extended vocal techniques on all 5 vowels. Existing
    singing voice datasets aim to capture a focused subset of singing
    voice characteristics, and generally consist of just a few singers.
    VocalSet contains recordings from 20 different singers (9 male, 11
    female) and a range of voice types.  VocalSet aims to improve the
    state of existing singing voice datasets and singing voice research by
    capturing not only a range of vowels, but also a diverse set of voices
    on many different vocal techniques, sung in contexts of scales,
    arpeggios, long tones, and excerpts.
    """

    def download(path):

        if path is None:
            path = os.environ["DATASET_path"]

        # Load the dataset (download if necessary) and set
        # the class attributes.

        t = time.time()

        if not os.path.isdir(path + "vocalset"):

            print("\tCreating Directory")
            os.mkdir(path + "vocalset")

        if not os.path.exists(path + "vocalset/VocalSet11.zip"):
            print("Downloading Vocalset")
            url = "https://zenodo.org/record/1442513/files/VocalSet11.zip?download=1"
            urllib.request.urlretrieve(url, path + "vocalset/VocalSet11.zip")

            print("vocalset downloaded in {} sec.".format(time.time() - t))

    def load(path=None):
        """
        Parameters
        ----------

        path: str (optional)
            a string where to load the data and download if not present

        Returns
        -------

        singers: list
            the list of singers as strings, 11 males and 9 females as in male1,
            male2, ...

        genders: list
            the list of genders of the singers as in male, male, female, ...

        vowels: list
            the vowels being pronunced

        data: list
            the list of waveforms, not all equal length

        """
        if path is None:
            path = os.environ["DATASET_PATH"]

        vocalset.download(path)
        t = time.time()

        # load wavs
        f = zipfile.ZipFile(path + "vocalset/VocalSet11.zip")

        # init. the data array
        singers = []
        genders = []
        vowels = []
        #        techniques = []
        data = []
        for filename in tqdm(f.namelist(), ascii=True):
            if ".wav" not in filename or "excerpts" in filename or "_" == filename[0]:
                continue
            vowel = filename[-5]
            if vowel not in ["a", "e", "i", "o", "u"]:
                continue
            vowels.append(vowel)
            bytes_ = io.BytesIO(f.read(filename))
            data.append(wav_read(bytes_)[1].astype("float32"))
            split = filename.split("/")
            genders.append("".join(x for x in split[1] if x.isalpha()))
            singers.append(split[1])
        #            techniques.append(split[-1][3:-6])

        return singers, genders, vowels, data
