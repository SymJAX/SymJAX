import os
import pickle,gzip
import urllib.request
import numpy as np
import tarfile
import time
from tqdm import tqdm
from scipy.io.wavfile import read as wav_read


class speech_commands:
    """music genre classification

    This dataset was used for the well known paper in genre classification
    "Musical genre classification of audio signals" by G. Tzanetakis
    and P. Cook in IEEE Transactions on Audio and Speech Processing 2002.

    Unfortunately the database was collected gradually and very early on in my
    research so I have no titles (and obviously no copyright permission etc).
    The files were collected in 2000-2001 from a variety of sources including
    personal CDs, radio, microphone recordings, in order to represent a variety
    of recording conditions. Nevetheless I have been providing it to researchers
    upon request mainly for comparison purposes etc. Please contact George
    Tzanetakis (gtzan@cs.uvic.ca) if you intend to publish experimental results
    using this dataset.

    There are some practical and conceptual issues with this dataset, described
    in "The GTZAN dataset: Its contents, its faults, their effects on
    evaluation, and its future use" by B. Sturm on arXiv 2013.
    """

    name2class = {'blues':0, 'classical':1, 'country': 2,
                  'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6,
                  'pop': 7, 'reggae': 8, 'rock': 9}
 
    def download(path=None):
        if path is None:
            path = os.environ['DATASET_path']
        path += 'speech_commands/'
        t0 = time.time()
    
        print('Downloading speech commands')
    
        # Check if directory exists
        if not os.path.isdir(path):
            print('\tCreating Directory')
            os.mkdir(path)
    
        # Check if file exists
        if not os.path.exists(path + 'speech_commands_v0.01.tar.gz'):
            url = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
            urllib.request.urlretrieve(url, path + 'speech_commands_v0.01.tar.gz')
    
    
    def load(path=None):
    
        if path is None:
            path = os.environ['DATASET_PATH']
        speech_commands.download(path)
    
        t0 = time.time()
    
        print('Loading speech command')
    
        tar = tarfile.open(path+'speech_commands/speech_commands_v0.01.tar.gz', 'r:gz')
    
        # Load train set
        wavs = list()
        labels = list()
        noises = list()
        noise_labels = list()
        names = tar.getmembers()
        for name in tqdm(names, ascii=True, total=1000):
            if 'wav' not in name.name:
                continue
            f = tar.extractfile(name.name)#.read()
            wav = wav_read(f)[1]
            if 'noise' in name.name:
                noises.append(wav)
                noise_labels.append(name.name.split('/')[-1])
            else:
                left = 16000 - len(wav)
                to_pad = left // 2
                wavs.append(np.pad(wav, [[to_pad, left - to_pad]]))
                labels.append(name.name.split('/')[-1])
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        y = np.array([np.nonzero(label == unique_labels)
                        for label in labels])
    
        print('Dataset speech commands loaded in{0:.2f}s.'.format(time.time()-t0))
    
        return np.array(wavs), y, labels, noises, noise_labels
