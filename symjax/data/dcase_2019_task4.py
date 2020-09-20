import os
import numpy as np
import zipfile
from tqdm import tqdm
import soundfile as sf
from .utils import download_dataset
import io
from scipy.io.wavfile import read as wav_read

_urls = {
    "https://zenodo.org/record/2583796/files/Synthetic_dataset.zip?download=1": "Synthetic_dataset.zip"
}


def load(path=None):
    """synthetic data for polyphonic event detection

    Synthetic data for DCASE 2019 task 4

    Freesound dataset [1,2]: A subset of FSD is used as foreground sound events for the synthetic subset of the dataset for DCASE 2019 task 4. FSD is a large-scale, general-purpose audio dataset composed of Freesound content annotated with labels from the AudioSet Ontology [3].

    SINS dataset [4]: The derivative of the SINS dataset used for DCASE2018 task 5 is used as background for the synthetic subset of the dataset for DCASE 2019 task 4.
    The SINS dataset contains a continuous recording of one person living in a vacation home over a period of one week.
    It was collected using a network of 13 microphone arrays distributed over the entire home.
    The microphone array consists of 4 linearly arranged microphones.

    The synthetic set is composed of 10 sec audio clips generated with Scaper [5]. The foreground events are obtained from FSD. Each event audio clip was verified manually to ensure that the sound quality and the event-to-background ratio were sufficient to be used an isolated event. We also verified that the event was actually dominant in the clip and we controlled if the event onset and offset are present in the clip. Each selected clip was then segmented when needed to remove silences before and after the event and between events when the file contained multiple occurrences of the event class.

    License:

    All sounds comming from FSD are released under Creative Commons licences. Synthetic sounds can only be used for competition purposes until the full CC license list is made available at the end of the competition.

    Further information on dcase website.

    References:

    [1] F. Font, G. Roma & X. Serra. Freesound technical demo. In Proceedings of the 21st ACM international conference on Multimedia. ACM, 2013.
     [2] E. Fonseca, J. Pons, X. Favory, F. Font, D. Bogdanov, A. Ferraro, S. Oramas, A. Porter & X. Serra. Freesound Datasets: A Platform for the Creation of Open Audio Datasets. In Proceedings of the 18th International Society for Music Information Retrieval Conference, Suzhou, China, 2017.
    [3] Jort F. Gemmeke and Daniel P. W. Ellis and Dylan Freedman and Aren Jansen and Wade Lawrence and R. Channing Moore and Manoj Plakal and Marvin Ritter. Audio Set: An ontology and human-labeled dataset for audio events. In Proceedings IEEE ICASSP 2017, New Orleans, LA, 2017.
    [4] Gert Dekkers, Steven Lauwereins, Bart Thoen, Mulu Weldegebreal Adhana, Henk Brouckxon, Toon van Waterschoot, Bart Vanrumste, Marian Verhelst, and Peter Karsmakers.
    The SINS database for detection of daily activities in a home environment using an acoustic sensor network.
    In Proceedings of the Detection and Classification of Acoustic Scenes and Events 2017 Workshop (DCASE2017), 32–36. November 2017.

    [5] J. Salamon, D. MacConnell, M. Cartwright, P. Li, and J. P. Bello. Scaper: A library for soundscape synthesis and augmentation In IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), New Paltz, NY, USA, Oct. 2017.
    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, "dcase_2019_task4", _urls)

    zfile = zipfile.ZipFile(path + "dcase_2019_task4/" + "Synthetic_dataset.zip")

    # labels
    file = zfile.open("synthetic_dataset.csv")
    labels = np.genfromtxt(
        file, delimiter="\t", encoding=None, dtype=None, names=True
    ).view(np.recarray)

    # Simplify labels
    for label in labels:
        label.event_label = label.event_label.replace("_", " ")

    # Classes
    classes = list(np.unique(labels.event_label))

    wavs = []
    for filename in tqdm(labels.filename, ascii=True):
        wav = zfile.read("audio/train/synthetic/" + filename)
        byt = io.BytesIO(wav)
        wavs.append(wav_read(byt)[1].astype("float32"))

    data = {"wavs": np.array(wavs), "labels": labels}
    return data
