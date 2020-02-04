import os
import pickle,gzip
import urllib.request
import numpy as np
import time

#from ..utils import  DownloadProgressBar

def download(path):
    # Check if directory exists
    if not os.path.isdir(path+'irmas'):
        print('Creating mnist Directory')
        os.mkdir(path+'irmas')

    # Check if file exists
    if not os.path.exists(path+'irmas/IRMAS-TrainingData.zip'):
        td  = time.time()
        url = 'https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1'
        urllib.request.urlretrieve(url,path + 'irmas/IRMAS-TrainingData.zip')

    # Check if file exists
    if not os.path.exists(path+'irmas/IRMAS-TestingData-Part1.zip'):
        td  = time.time()
        url = 'https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip?download=1'
        urllib.request.urlretrieve(url,path + 'irmas/IRMAS-TestingData-Part1.zip')

    # Check if file exists
    if not os.path.exists(path+'irmas/IRMAS-TestingData-Part2.zip'):
        td  = time.time()
        url = 'https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip?download=1'
        urllib.request.urlretrieve(url,path + 'irmas/IRMAS-TestingData-Part2.zip')

    # Check if file exists
    if not os.path.exists(path+'irmas/IRMAS-TestingData-Part3.zip'):
        td  = time.time()
        url = 'https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip?download=1'
        urllib.request.urlretrieve(url,path + 'irmas/IRMAS-TestingData-Part3.zip')




def load(PATH=None):
    """IRMAS.
    ref https://zenodo.org/record/1290750#.WzCwSRyxXMU

    Parameters
    ----------
        path: str (optional)
            default $DATASET_PATH), the path to look for the data and
            where the data will be downloaded if not present
    """

    if PATH is None:
        PATH = os.environ['DATASET_PATH']
    download(PATH)

    t0 = time.time()

    wavs = list()
    labels = list()

    f       = zipfile.ZipFile(PATH+'irmas/IRMAS-TrainingData.zip')
    namelist = f.namelist()
    for filename in tqdm(namelist, ascii=True):
        if '.wav' not in filename:
            continue
        wavfile = f.read(filename)
        byt = io.BytesIO(wavfile)
        wavs.append(wav_read(byt)[1].astype('float32'))
        labels.append(filename.split('/')[-2])

    for part in ['1', '2', '3']:
        f = zipfile.ZipFile(PATH+'irmas/IRMAS-TestingData-Part{}.zip'.format(part))
        namelist = f.namelist()
        for filename in tqdm(namelist, ascii=True):
            if '.wav' not in filename:
                continue

            byt = io.BytesIO(f.read(filename))
            wavs.append(wav_read(byt)[1].astype('float32'))

            byt = io.BytesIO(f.read(filename.replace('.wav', '.txt')))
            labels.append(np.loadtxt(byt, dtype='str')[0])

    categories = np.array(labels)
    labels = np.zeros(len(categories))
    wavs = np.array(wavs)
    for i, c in enumerate(np.unique(categories)):
        labels[categories == c] = i

    return wavs, labels, categories






    print('Dataset mnist loaded in {0:.2f}s.'.format(time.time()-t0))

    return train_set, valid_set, test_set

