import os
import h5py
import numpy as np
import tarfile
import time
from .utils import download_dataset

_dataset = "camelyonpatch"

DOC = """Audio binary classification, presence or absence of bird songs.
    `freefield1010 <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_.
    is a collection of over 7,000 excerpts from field recordings
    around the world, gathered by the FreeSound project, and then standardised
    for research. This collection is very diverse in location and environment,
    and for the BAD Challenge we have newly annotated it for the
    presence/absence of birds.
    """

_baseurl = "https://zenodo.org/record/2546921/files/"
_urls = {
    "camelyonpatch_level_2_split_test_meta.csv?download=1": "test_meta.csv",
    "camelyonpatch_level_2_split_test_x.h5.gz?download=1": "test_x.h5.gz",
    "camelyonpatch_level_2_split_test_y.h5.gz?download=1": "test_y.h5.gz",
    "camelyonpatch_level_2_split_train_meta.csv?download=1": "train_meta.csv",
    "camelyonpatch_level_2_split_train_x.h5.gz?download=1": "train_x.h5.gz",
    "camelyonpatch_level_2_split_train_y.h5.gz?download=1": "train_y.h5.gz",
    "camelyonpatch_level_2_split_valid_meta.csv?download=1": "valid_meta.csv",
    "camelyonpatch_level_2_split_valid_x.h5.gz?download=1": "valid_x.h5.gz",
    "camelyonpatch_level_2_split_valid_y.h5.gz?download=1": "valid_y.h5.gz",
}


def load(path=None):
    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, _dataset, _urls, _baseurl)

    t = time.time()

    for file in _urls:
        print(_urls[file])
        if _urls[file][-2:] == "gz":
            tar = tarfile.open(os.path.join(path, _dataset, _urls[file]), "r:gz")
            for member in tar.getmembers():
                print(member)
                f = tar.extractfile(member)
                if f is not None:
                    with h5py.File(f, "r") as g:
                        print("Keys: %s" % f.keys())
                        a_group_key = list(f.keys())[0]
                    asdf

    return wavs, labels
