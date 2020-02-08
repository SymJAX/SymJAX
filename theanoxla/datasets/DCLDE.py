import os
import gzip
import urllib.request
import numpy as np
import time
import zipfile
import io
from scipy.io.wavfile import read as wav_read
from tqdm import tqdm


def download(path):
    """ToDo
    """

    # Load the dataset (download if necessary) and set
    # the class attributes.

    print("Loading DCLDE")
    t = time.time()

    if not os.path.isdir(path+'DCLDE'):
        print('\tCreating Directory')
        os.mkdir(path+'DCLDE')
    if not os.path.exists(path+'DCLDE/DCLDE_LF_Dev.zip'):
        url = 'http://sabiod.univ-tln.fr/workspace/DCLDE2018/DCLDE_LF_Dev.zip'
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
                                                    desc='Wav files') as t:
            urllib.request.urlretrieve(url,path+'DCLDE/DCLDE_LF_Dev.zip')



def load(window_size=441000,path=None):
    """ToDo
    """
    if path is None:
        path = os.environ['DATASET_path']
    download(path)

    # Loading the files
    f       = zipfile.ZipFile(path+'DCLDE/DCLDE_LF_Dev.zip')
    wavs    = list()
#    labels  = list()
    for zipf in tqdm(f.filelist,ascii=True):
        if '.wav' in zipf.filename and '.d100.' in zipf.filename:
            wavfile   = f.read(zipf)
            byt       = io.BytesIO(wavfile)
            wav       = wav_read(byt)[1].astype('float32')
            for s in range(len(wav)//window_size):
                wavs.append(wav[s*window_size:(s+1)*window_size])
#            labels.append(zipf.filename.split('/')[2])
#    return wavs,labels
    wavs = np.expand_dims(np.asarray(wavs),1)
    dataset.add_variable({'signals':{'train_set':wavs}})

    print('Dataset freefield1010 loaded in','{0:.2f}'.format(time.time()-t),'s.')
    return dataset


