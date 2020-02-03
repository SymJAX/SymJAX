import os
import pickle,gzip
import urllib.request
import numpy as np
import time
import tarfile
from tqdm import tqdm
from scipy.io.wavfile import read as wav_read


def download(path):

    # Check if directory exists
    if not os.path.isdir(path+'ust'):
        print('Creating mnist Directory')
        os.mkdir(path+'ust')

    url = 'https://zenodo.org/record/3233082/files/audio-dev.tar.gz?download=1'
    # Check if file exists
    if not os.path.exists(path+'ust/audio-dev.tar.gz'):
        td  = time.time()
        urllib.request.urlretrieve(url, path+'ust/audio-dev.tar.gz')

    url = 'https://zenodo.org/record/3233082/files/annotations-dev.csv?download=1'
    # Check if file exists
    if not os.path.exists(path+'ust/annotations-dev.csv'):
        td  = time.time()
        urllib.request.urlretrieve(url, path+'ust/annotations-dev.csv')




def load(path=None, classes=range(10)):
    """Urban.
    Reference at https://zenodo.org/record/3233082

    Parameters
    ----------
        path: str (optional)
            default $DATASET_path), the path to look for the data and
            where the data will be downloaded if not present
    """

    if path is None:
        path = os.environ['DATASET_PATH']
    download(path)

    t0 = time.time()

    # Loading the file
    files = tarfile.open(path+'ust/audio-dev.tar.gz', 'r:gz')
    annotations = np.loadtxt(path+'ust/annotations-dev.csv', delimiter=',',
                             skiprows=1, dtype='str')

    # get name
    filenames = list(annotations[:, 2])
    for i in range(len(filenames)):
        filenames[i] = annotations[i, 0] + '/' + str(filenames[i])

    # get fine labels and limts for coarse classes
    fine_labels = annotations[:, 4: 33].astype('float32').astype('int32')
    class_limits = [0, 4, 9, 10, 14, 19, 23, 28, 29]
    n_classes = len(class_limits) - 1
    n_samples = len(annotations)
    llabels = np.zeros((n_samples, n_classes), dtype='int')
    for k in range(n_classes):
        block = fine_labels[:, class_limits[k]: class_limits[k+1]]
        block = block.astype('float32').astype('int32')
        llabels[:, k] = block.max(1)

    POT = []
    wavs = np.zeros((2794, 441000))
    labels = np.zeros((2794, n_classes)).astype('int')
    filenames = files.getnames()
    cpt = 0
    for name in tqdm(filenames, ascii=True):
        if '.wav' not in name:
            continue
        wav = wav_read(files.extractfile(name))[1].astype('float32')
        wavs[cpt, :len(wav)] = wav
        labels[cpt] = llabels[filenames.index(name)]
        cpt += 1
    return wavs, labels
