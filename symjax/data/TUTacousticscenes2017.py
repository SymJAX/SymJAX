import os
import numpy as np
import zipfile
from tqdm import tqdm
import soundfile as sf
from .utils import download_dataset


_urls = {
    "https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.audio.{}.zip".format(
        u
    ): "TUT-acoustic-scenes-2017-development.audio.{}.zip".format(
        u
    )
    for u in range(1, 11)
}


_urls.update(
    {
        "https://zenodo.org/record/1040168/files/TUT-acoustic-scenes-2017-evaluation.audio.{}.zip".format(
            u
        ): "TUT-acoustic-scenes-2017-evaluation.audio.{}.zip".format(
            u
        )
        for u in range(1, 5)
    }
)


_urls.update(
    {
        "https://zenodo.org/record/400515/files/TUT-acoustic-scenes-2017-development.meta.zip": "TUT-acoustic-scenes-2017-development.meta.zip"
    }
)

_urls.update(
    {
        "https://zenodo.org/record/1040168/files/TUT-acoustic-scenes-2017-evaluation.meta.zip": "TUT-acoustic-scenes-2017-evaluation.meta.zip"
    }
)


classes = [
    "metro_station",
    "office",
    "park",
    "residential_area",
    "train",
    "tram",
    "library",
    "home",
    "grocery_store",
    "forest_path",
    "city_center",
    "car",
    "cafe/restaurant",
    "bus",
    "beach",
]


def load(path=None):
    """Acoustic Scene classification

    TUT Acoustic Scenes 2017 dataset consists of recordings from various
    acoustic scenes, all having distinct recording locations. For each
    recording location, 3-5 minute long audio recording was captured.
    The original recordings were then split into segments with a length of 10
    seconds. These audio segments are provided in individual files.

    Roughly 11Go of data.

    Acoustic scenes for the task (15):

        Bus - traveling by bus in the city (vehicle)
        Cafe / Restaurant - small cafe/restaurant (indoor)
        Car - driving or traveling as a passenger, in the city (vehicle)
        City center (outdoor)
        Forest path (outdoor)
        Grocery store - medium size grocery store (indoor)
        Home (indoor)
        Lakeside beach (outdoor)
        Library (indoor)
        Metro station (indoor)
        Office - multiple persons, typical work day (indoor)
        Residential area (outdoor)
        Train (traveling, vehicle)
        Tram (traveling, vehicle)
        Urban park (outdoor)
    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, "irmas", _urls)

    # meta data
    filename = (
        path + "TUTacousticscences2017/TUT-acoustic-scenes-2017-development.meta.zip"
    )
    meta = zipfile.ZipFile(filename)
    meta_names = list()
    wav_names = list()
    wav_labels = list()
    for filename in meta.namelist():
        if "train" not in filename and "evaluate" not in filename:
            continue
        meta_names.append(filename.split("/")[-1][:-4])
        content = np.loadtxt(
            io.BytesIO(meta.read(filename)), delimiter="\t", dtype="str"
        )
        wav_names.append(list(content[:, 0]))
        wav_labels.append(list(content[:, 1]))
        for j in range(len(wav_names[-1])):
            wav_names[-1][j] = wav_names[-1][j].split("/")[-1]

    folds = list()
    wavs = list()
    labels = list()

    filename = (
        path
        + "TUTacousticscences2017/"
        + "TUT-acoustic-scenes-2017-development.audio.{}.zip"
    )

    for part in range(1, 11):
        # load wavs
        f = zipfile.ZipFile(filename.format(part))
        for name in tqdm(
            f.namelist(), ascii=True, desc="Train Part:{}/10".format(part)
        ):
            if ".wav" not in name:
                continue
            wavfile = f.read(name)
            byt = io.BytesIO(wavfile)
            wavs.append(sf.read(byt)[0].astype("float32"))
            nn = name.split("/")[-1]
            add_label = 1
            folds.append([])
            for meta_name, wav_name, wav_label in zip(
                meta_names, wav_names, wav_labels
            ):
                if nn in wav_name:
                    if add_label:
                        index = wav_name.index(nn)
                        labels.append(wav_label[index])
                        add_label = 0
                    folds[-1].append(meta_name)

    # now format the folds
    train_folds = np.zeros((len(folds), 4), dtype="bool")
    for i in range(len(folds)):
        for fold in folds[i]:
            if "train" in fold:
                train_folds[i, int(fold.split("_")[0][-1]) - 1] = 1

    # now deal with the test set

    # meta data
    filename = (
        path
        + "TUTacousticscences2017/"
        + "TUT-acoustic-scenes-2017-evaluation.meta.zip"
    )
    meta = zipfile.ZipFile(filename)
    for filename in meta.namelist():
        if "evaluate.txt" in filename:
            targets = np.loadtxt(
                io.BytesIO(meta.read(filename)),
                delimiter="\t",
                dtype="str",
            )

    targets_0 = [c.split("/")[1] for c in targets[:, 0]]
    targets_1 = list(targets[:, 1])

    mapping = dict(zip(targets_0, targets_1))

    test_wavs = list()
    test_labels = list()
    filename = (
        path + "TUTacousticscences2017/TUT-acoustic-scenes-2017-evaluation.audio.{}.zip"
    )

    for part in range(1, 5):
        # load wavs
        f = zipfile.ZipFile(filename.format(part))
        for name in tqdm(
            f.namelist(),
            ascii=True,
            desc="Test Data Part{}/4".format(part),
        ):
            if ".wav" not in name:
                continue
            byt = io.BytesIO(f.read(name))
            test_wavs.append(sf.read(byt)[0].astype("float32"))
            test_labels.append(mapping[name.split("/")[-1]])

    # turn the labels into integers
    mapping = dict(zip(classes, list(range(15))))
    labels = [mapping[l] for l in labels]
    test_labels = [mapping[l] for l in test_labels]

    data = {
        "train_set/wavs": np.array(wavs),
        "train_set/labels": np.array(labels),
        "test_set/wavs": np.array(test_wavs),
        "test_set/labels": np.array(test_labels),
        "folds": train_folds,
    }
    return data
