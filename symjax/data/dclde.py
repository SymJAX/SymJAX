import io
import os
import time
import urllib.request
import zipfile

import numpy as np
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm


class dclde:
    """
    The high-frequency dataset consists of marked encounters with echolocation
    clicks of species commonly found along the US Atlantic Coast, and in the
    Gulf of Mexico:

    Mesoplodon europaeus - Gervais' beaked whale
    Ziphius cavirostris - Cuvier's beaked whale
    Mesoplodon bidens - Sowerby's beaked whale
    Lagenorhynchus acutus - Atlantic white-sided dolphin
    Grampus griseus - Risso's dolphin
    Globicephala macrorhynchus - Short-finned pilot whale
    Stenella sp. - Stenellid dolphins
    Delphinid type A
    Delphinid type B
    Unidentified delphinid - delphinid other than those described above

    The goal for these datasets is to identify acoustic encounters by species
    during times when animals were echolocating. Analysts examined data for
    echolocation clicks and approximated the start and end times of acoustic
    encounters. Any period that was separated from another one by five minutes
    or more was marked as a separate encounter. Whistle activity was not
    considered. Consequently, while the use of whistle information during
    echolocation activity is appropriate, reporting a species based on whistles
    in the absence of echolocation activity will be considered a false positive
    for this classification task.
    """

    def download(path):
        """ToDo"""

        # Load the dataset (download if necessary) and set
        # the class attributes.

        print("Loading DCLDE")
        t = time.time()

        if not os.path.isdir(path + "DCLDE"):
            print("\tCreating Directory")
            os.mkdir(path + "DCLDE")
        if not os.path.exists(path + "DCLDE/DCLDE_LF_Dev.zip"):
            url = "http://sabiod.univ-tln.fr/workspace/DCLDE2018/DCLDE_LF_Dev.zip"
            with DownloadProgressBar(
                unit="B", unit_scale=True, miniters=1, desc="Wav files"
            ) as t:
                urllib.request.urlretrieve(url, path + "DCLDE/DCLDE_LF_Dev.zip")

    def load(window_size=441000, path=None):
        """ToDo"""
        if path is None:
            path = os.environ["DATASET_path"]
        dclde.download(path)

        # Loading the files
        f = zipfile.ZipFile(path + "DCLDE/DCLDE_LF_Dev.zip")
        wavs = list()
        #    labels  = list()
        for zipf in tqdm(f.filelist, ascii=True):
            if ".wav" in zipf.filename and ".d100." in zipf.filename:
                wavfile = f.read(zipf)
                byt = io.BytesIO(wavfile)
                wav = wav_read(byt)[1].astype("float32")
                for s in range(len(wav) // window_size):
                    wavs.append(wav[s * window_size : (s + 1) * window_size])
        #            labels.append(zipf.filename.split('/')[2])
        #    return wavs,labels
        wavs = np.expand_dims(np.asarray(wavs), 1)
        dataset.add_variable({"signals": {"train_set": wavs}})

        print(
            "Dataset freefield1010 loaded in", "{0:.2f}".format(time.time() - t), "s."
        )
        return dataset
