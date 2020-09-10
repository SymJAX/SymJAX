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

fine_to_coarse = {
    "dog": 0,
    "rooster": 0,
    "pig": 0,
    "cow": 0,
    "frog": 0,
    "cat": 0,
    "hen": 0,
    "insects": 0,
    "sheep": 0,
    "crow": 0,
    "rain": 1,
    "sea_waves": 1,
    "crackling_fire": 1,
    "crickets": 1,
    "chirping_birds": 1,
    "water_drops": 1,
    "wind": 1,
    "pouring_water": 1,
    "toilet_flush": 1,
    "thunderstorm": 1,
    "crying_baby": 2,
    "sneezing": 2,
    "clapping": 2,
    "breathing": 2,
    "coughing": 2,
    "footsteps": 2,
    "laughing": 2,
    "brushing_teeth": 2,
    "snoring": 2,
    "drinking_sipping": 2,
    "door_wood_knock": 3,
    "mouse_click": 3,
    "keyboard_typing": 3,
    "door_wood_creaks": 3,
    "can_opening": 3,
    "washing_machine": 3,
    "vacuum_cleaner": 3,
    "clock_alarm": 3,
    "clock_tick": 3,
    "glass_breaking": 3,
    "helicopter": 4,
    "chainsaw": 4,
    "siren": 4,
    "car_horn": 4,
    "engine": 4,
    "train": 4,
    "church_bells": 4,
    "airplane": 4,
    "fireworks": 4,
    "hand_saw": 4,
}

_urls = {
    "https://github.com/karoldvl/ESC-50/archive/master.zip": "master.zip",
}


def load(path=None):
    """ESC-10/50: Environmental Sound Classification

    https://github.com/karolpiczak/ESC-50#download

    The ESC-50 dataset is a labeled collection of 2000 environmental audio
    recordings suitable for benchmarking methods of environmental sound
    classification.

    The dataset consists of 5-second-long recordings organized into 50
    semantical classes (with 40 examples per class) loosely arranged into 5
    major categories:
        Animals
        Natural soundscapes & water sounds
        Human, non-speech sounds
        Interior/domestic sounds
        Exterior/urban noises

    Clips in this dataset have been manually extracted from public field
    recordings gathered by the Freesound.org project. The dataset has been
    prearranged into 5 folds for comparable cross-validation, making sure
    that fragments from the same original source file are contained in a
    single fold.

    ESC 50.

    https://github.com/karolpiczak/ESC-50#download


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

    fine_labels: array
        the labels of the final classes (50 different ones) as a integer
        vector

    coarse_labels: array
        the labels of the classes big cateogry (5 of them)

    folds: array
        the fold as an integer from 1 to 5 specifying how to split the data
        one should not split a fold into train and set as it would
        make the same recording (but different subparts) be present in train
        and test, biasing optimistically the results.

    esc10: array
        the boolean vector specifying if the corresponding datum (wav, label,
        ...) is in the ESC-10 dataset or not. That is, to load the ESC-10
        dataset simply load ESC-50 and use this boolean vector to extract
        only the ESC-10 data.
    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, _dataset, _urls, _baseurl)

    t0 = time.time()

    f = zipfile.ZipFile(path + "esc50/master.zip")

    meta = np.loadtxt(
        io.BytesIO(f.read("ESC-50-master/meta/esc50.csv")),
        delimiter=",",
        skiprows=1,
        dtype="str",
    )
    filenames = list(meta[:, 0])
    folds = meta[:, 1].astype("int32")
    fine_labels = meta[:, 2].astype("int32")
    categories = meta[:, 3]
    esc10 = meta[:, 4] == "True"
    coarse_labels = np.array([esc.fine_to_coarse[c] for c in categories])
    coarse_labels = coarse_labels.astype("int32")

    wavs = list()
    order = list()
    N = 0
    for filename in tqdm(f.namelist(), ascii=True):
        if ".wav" not in filename:
            continue
        wavfile = f.read(filename)
        byt = io.BytesIO(wavfile)
        wavs.append(wav_read(byt)[1].astype("float32"))
        order.append(filenames.index(filename.split("/")[-1]))
        N = max(N, len(wavs[-1]))

    all_wavs = np.zeros((len(wavs), N))
    for i in range(len(wavs)):
        left = (N - len(wavs[i])) // 2
        all_wavs[order[i], left : left + len(wavs[i])] = wavs[i]
    data = {
        "wavs": all_wavs,
        "fine_labels": fine_labels,
        "coarse_labels": coarse_labels,
        "folds": folds,
        "esc10": esc10,
    }
    return data
