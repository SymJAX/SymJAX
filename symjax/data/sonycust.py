import os
import numpy as np
import tarfile
from tqdm import tqdm
from scipy.io.wavfile import read as wav_read
from .utils import download_dataset


_urls = {
    "https://zenodo.org/record/3233082/files/audio-dev.tar.gz?download=1": "audio-dev.tar.gz",
    "https://zenodo.org/record/3233082/files/annotations-dev.csv?download=1": "annotations-dev.csv",
}


coarse_labels = [
    "engine",
    "machinery-impact",
    "non-machinery-impact" "powered-saw",
    "alert-signal",
    "music",
    "human-voice",
    "dog",
]

fine_labels = [
    "small-sounding-engine",
    "medium-sounding-engine",
    "large-sounding-engine",
    "engine-of-uncertain-size",
    "rock-drill",
    "jackhammer",
    "hoe-ram",
    "pile-driver",
    "other-unknown-impact-machinery",
    "non-machinery-impact",
    "chainsaw",
    "small-medium-rotating-saw",
    "large-rotating-saw",
    "other-unknown-powered-saw",
    "car-horn",
    "car-alarm",
    "siren",
    "reverse-beeper",
    "other-unknown-alert-signal",
    "stationary-music",
    "mobile-music",
    "ice-cream-truck",
    "music-from-uncertain-source",
    "person-or-small-group-talking",
    "person-or-small-group-shouting",
    "large-crowd",
    "amplified-speech",
    "other-unknown-human-voice",
    "dog-barking-whining",
]


def load(path=None):
    """multilabel urban sound classification

    Reference at https://zenodo.org/record/3233082

    Description

    SONYC Urban Sound Tagging (SONYC-UST) is a dataset for the development and
    evaluation of machine listening systems for realistic urban noise
    monitoring. The audio was recorded from the SONYC acoustic sensor network.
    Volunteers on the Zooniverse citizen science platform tagged the presence
    of 23 classes that were chosen in consultation with the New York City
    Department of Environmental Protection. These 23 fine-grained classes can be
    grouped into 8 coarse-grained classes. The recordings are split into three
    subsets: training, validation, and test. These sets are disjoint with
    respect to the sensor from which each recording came. For increased
    reliability, three volunteers annotated each recording, and members of the
    SONYC team subsequently created a set of ground-truth tags for the
    validation set using a two-stage annotation procedure in which two
    annotators independently tagged and then collectively resolved any
    disagreements. For more details on the motivation and creation of this
    dataset see the DCASE 2019 Urban Sound Tagging Task website.

    Audio data

    The provided audio has been acquired using the SONYC acoustic sensor network
    for urban noise pollution monitoring. Over 50 different sensors have been
    deployed in New York City, and these sensors have collectively gathered the
    equivalent of 37 years of audio data, of which we provide a small subset.
    The data was sampled by selecting the nearest neighbors on VGGish features
    of recordings known to have classes of interest. All recordings are 10
    seconds and were recorded with identical microphones at identical gain
    settings. To maintain privacy, the recordings in this release have been
    distributed in time and location, and the time and location of the
    recordings are not included in the metadata.

    Labels

    there are fine and coarse labels
    engine
    1: small-sounding-engine
    2: medium-sounding-engine
    3: large-sounding-engine
    X: engine-of-uncertain-size
    machinery-impact
    1: rock-drill
    2: jackhammer
    3: hoe-ram
    4: pile-driver
    X: other-unknown-impact-machinery
    non-machinery-impact
    1: non-machinery-impact
    powered-saw
    1: chainsaw
    2: small-medium-rotating-saw
    3: large-rotating-saw
    X: other-unknown-powered-saw
    alert-signal
    1: car-horn
    2: car-alarm
    3: siren
    4: reverse-beeper
    X: other-unknown-alert-signal
    music
    1: stationary-music
    2: mobile-music
    3: ice-cream-truck
    X: music-from-uncertain-source
    human-voice
    1: person-or-small-group-talking
    2: person-or-small-group-shouting
    3: large-crowd
    4: amplified-speech
    X: other-unknown-human-voice
    dog
    1: dog-barking-whining

    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, "irmas", _urls)

    # Loading the file
    files = tarfile.open(path + "ust/audio-dev.tar.gz", "r:gz")
    annotations = np.loadtxt(
        path + "ust/annotations-dev.csv",
        delimiter=",",
        skiprows=1,
        dtype="str",
    )

    # get name
    filenames = list(annotations[:, 2])
    for i in range(len(filenames)):
        filenames[i] = annotations[i, 0] + "/" + str(filenames[i])

    # get fine labels and limts for coarse classes
    fine_labels = annotations[:, 4:33].astype("float32").astype("int32")
    class_limits = [0, 4, 9, 10, 14, 19, 23, 28, 29]
    n_classes = len(class_limits) - 1
    n_samples = len(annotations)
    llabels = np.zeros((n_samples, n_classes), dtype="int")
    for k in range(n_classes):
        block = fine_labels[:, class_limits[k] : class_limits[k + 1]]
        llabels[:, k] = block.max(1)

    wavs = np.zeros((2794, 441000), dtype="float32")
    coarse = np.zeros((2794, 8), dtype="int32")
    fine = np.zeros((2794, 29), dtype="int32")
    filenames = files.getnames()
    cpt = 0
    for name in tqdm(filenames, ascii=True):
        if ".wav" not in name:
            continue
        wavs[cpt] = wav_read(files.extractfile(name))[1].astype("float32")
        coarse[cpt] = llabels[filenames.index(name)]
        fine[cpt] = fine_labels[filenames.index(name)]
        cpt += 1

    data = {"wavs": wavs, "fine_labels": fine, "coarse_labels": coarse}
    return data
