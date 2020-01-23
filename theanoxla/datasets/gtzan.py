import os
import pickle,gzip
import urllib.request
import numpy as np
import tarfile
import time
from scipy.io.wavfile import read as wav_read


def download(PATH=None):
    if PATH is None:
        PATH = os.environ['DATASET_PATH']

    t0 = time.time()

    print('Downloading gtzan')

    # Check if directory exists
    if not os.path.isdir(PATH+'gtzan'):
        print('\tCreating Directory')
        os.mkdir(PATH+'gtzan')

    # Check if file exists
    if not os.path.exists(PATH+'gtzan//genres.tar.gz'):
        url = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'
        urllib.request.urlretrieve(url,PATH+'gtzan/genres.tar.gz')


def load(subsample=1, PATH=None):

    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    download(PATH)

    t0 = time.time()

    print('Loading gtzan')

    tar = tarfile.open(PATH+'gtzan/genres.tar.gz', 'r:gz')

    # Load train set
    train_songs = list()
    train_labels = list()
    names = tar.getnames()
    name2class = {'blues':0, 'classical':1, 'country': 2,
                  'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
                  'pop': 7, 'reggae': 8, 'rock': 9}
    names = tar.getmembers()
    for name in names:
        if 'wav' not in name.name:
            continue
        f = tar.extractfile(name.name)#.read()
        train_songs.append(wav_read(f)[1][::subsample])
        t = name.name.split('/')[1]
        train_labels.append(name2class[t])
        print(train_songs[-1].shape)

    N = np.min([len(w) for w in train_songs])
    train_songs = [w[:N] for w in train_songs]

    train_songs = np.stack(train_songs).astype('float32')
    train_labels = np.array(train_labels).astype('int32')

    print('Dataset gtzan loaded in{0:.2f}s.'.format(time.time()-t0))

    return train_songs, train_labels
