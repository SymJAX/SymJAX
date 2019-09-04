import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm

from ..utils import to_one_hot, DownloadProgressBar


from . import Dataset

def load_freefield1010(subsample=1,n_samples=-1,PATH=None):
    """Audio binary classification, presence or absence of bird songs.
    `freefield1010 <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_. 
    is a collection of over 7,000 excerpts from field recordings 
    around the world, gathered by the FreeSound project, and then standardised 
    for research. This collection is very diverse in location and environment, 
    and for the BAD Challenge we have newly annotated it for the 
    presence/absence of birds.

    Parameters
    ----------

    subsample : int (optional)
        the amount of subsample ot apply. No low-pass filtering is applied 
        before-hand; too much subsample will create aliasing. Default value
        does not subsample the signals.

    n_samples : int (optional)
        The number of samples to load. By default, load the whole dataset.

    PATH : str (optional)
        The path to use for dataset loading (downloading if needed). By
        default use the environment variable :envvar:`DATASET_PATH`.
    """
    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    dict_init = [('sampling_rate',44100),("n_classes",2),("path",PATH),
                ("name","freefield1010"),('classes',["no bird","bird"])]
    dataset = Dataset(**dict(dict_init))

    # Load the dataset (download if necessary) and set
    # the class attributes.
        
    print("Loading freefield1010")
    t = time.time()
        
    if not os.path.isdir(PATH+'freefield1010'):
        print('\tCreating Directory')
        os.mkdir(PATH+'freefield1010')

    if not os.path.exists(PATH+'freefield1010/ff1010bird_wav.zip'):
        url = 'https://archive.org/download/ff1010bird/ff1010bird_wav.zip'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                                    desc='Wav files') as t:
            urllib.request.urlretrieve(url,PATH+
                    'freefield1010/ff1010bird_wav.zip')  

    if not os.path.exists(PATH+'freefield1010/ff1010bird_metadata.csv'):
        url = 'https://ndownloader.figshare.com/files/6035814'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                        desc='Metadata') as t:
            urllib.request.urlretrieve(url,PATH+
                                'freefield1010/ff1010bird_metadata.csv')  

    #Loading Labels
    labels = np.loadtxt(PATH+'freefield1010/ff1010bird_metadata.csv',
            delimiter=',',skiprows=1,dtype='int32')
    if n_samples<len(labels):
        labels = labels[np.random.permutation(len(labels))[:n_samples]]
    # Loading the files
    f       = zipfile.ZipFile(PATH+'freefield1010/ff1010bird_wav.zip')
    # Init. the wavs matrix
    N       = labels.shape[0]
    wavs    = np.empty((N,441000//subsample),dtype='float32')
    for i,files_ in tqdm(enumerate(labels[:,0]),ascii=True):
        wavfile   = f.read('wav/'+str(files_)+'.wav')
        byt       = io.BytesIO(wavfile)
        wavs[i]   = wav_read(byt)[1].astype('float32')[::subsample]

    labels = labels[:,1]
    wavs   = np.expand_dims(wavs,1)

    dataset['signals/train_set']=wavs
    dataset['labels/train_set']=labels

    return dataset


