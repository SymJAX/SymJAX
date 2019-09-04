import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read

from . import Dataset
from tqdm import tqdm

from ..utils import to_one_hot, DownloadProgressBar


def load_warblr(PATH=None):
    """Binary audio classification, presence or absence of a bird.
    `Warblr <http://machine-listening.eecs.qmul.ac.uk/bird-audio-detection-challenge/#downloads>`_ 
    comes from a UK bird-sound crowdsourcing 
    research spinout called Warblr. From this initiative we have 
    10,000 ten-second smartphone audio recordings from around the UK. 
    The audio totals around 44 hours duration. The audio will be 
    published by Warblr under a Creative Commons licence. The audio 
    covers a wide distribution of UK locations and environments, and 
    includes weather noise, traffic noise, human speech and even human 
    bird imitations. It is directly representative of the data that is 
    collected from a mobile crowdsourcing initiative.

    :param data_format: (optional, default 'NCHW')
    :type data_format: 'NCHW' or 'NHWC'
    :param path: (optional, default $DATASET_PATH), the path to look for the data and 
                     where the data will be downloaded if not present
    :type path: str
    """
    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    dict_init = [('sampling_rate',44100),("n_classes",2),("path",PATH),
                ("name","warblr"),('classes',["no bird","bird"])]

    dataset = Dataset(**dict(dict_init))
        
    # Load the dataset (download if necessary) and set
    # the class attributes.
        
    print('Loading warblr')
    t = time.time()
    if not os.path.isdir(PATH+'warblr'):
        print('\tCreating Directory')
        os.mkdir(PATH+'warblr')

    if not os.path.exists(PATH+'warblr/warblrb10k_public_wav.zip'):
        url = 'https://archive.org/download/warblrb10k_public/warblrb10k_public_wav.zip'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                        desc='Downloading dataset') as t:

            urllib.request.urlretrieve(url,PATH+\
                                'warblr/warblrb10k_public_wav.zip')
        
    if not os.path.exists(PATH+'warblr/warblrb10k_public_metadata.csv'):
        url = 'https://ndownloader.figshare.com/files/6035817'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                            desc='Downloading metadata') as t:

            urllib.request.urlretrieve(url,PATH+\
                                'warblr/warblrb10k_public_metadata.csv')  

    #Loading Labels
    labels = np.loadtxt(PATH+'warblr/warblrb10k_public_metadata.csv',
            delimiter=',',skiprows=1,dtype='str')
    # Loading the files
    f    = zipfile.ZipFile(PATH+'warblr/warblrb10k_public_wav.zip')
    N    = labels.shape[0]
    wavs = list()
    for i,files_ in tqdm(enumerate(labels),ascii=True):
        wavfile   = f.read('wav/'+files_[0]+'.wav')
        byt       = io.BytesIO(wavfile)
        wavs.append(np.expand_dims(wav_read(byt)[1].astype('float32'),0))
    labels    = labels[:,1].astype('int32')
    dataset.add_variable({'signals':[{'train_set':wavs},
                                    (1,None),'float32'],
                        'labels':[{'train_set':labels},
                                    (),'int32']})

    print('Dataset warblr loaded in','{0:.2f}'.format(time.time()-t),'s.')
    return dataset

