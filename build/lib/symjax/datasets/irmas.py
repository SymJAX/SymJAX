import os
import pickle,gzip
import urllib.request
import numpy as np
import time

#from ..utils import  DownloadProgressBar
class irmas:
    """music instrument classification

    ref https://zenodo.org/record/1290750#.WzCwSRyxXMU

    This dataset includes musical audio excerpts with annotations of the
    predominant instrument(s) present. It was used for the evaluation in the
    following article:

    Bosch, J. J., Janer, J., Fuhrmann, F., & Herrera, P. “A Comparison of Sound
    Segregation Techniques for Predominant Instrument Recognition in Musical
    Audio Signals”, in Proc. ISMIR (pp. 559-564), 2012

    Please Acknowledge IRMAS in Academic Research

    IRMAS is intended to be used for training and testing methods for the
    automatic recognition of predominant instruments in musical audio. The
    instruments considered are: cello, clarinet, flute, acoustic guitar,
    electric guitar, organ, piano, saxophone, trumpet, violin, and human singing
    voice. This dataset is derived from the one compiled by Ferdinand Fuhrmann
    in his PhD thesis, with the difference that we provide audio data in stereo
    format, the annotations in the testing dataset are limited to specific
    pitched instruments, and there is a different amount and lenght of excerpts.
    """

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
    
    
    
    
    def load(path=None):
    
        if path is None:
            path = os.environ['DATASET_PATH']
        irmas.download(path)
    
        t0 = time.time()
    
        wavs = list()
        labels = list()
    
        f       = zipfile.ZipFile(path+'irmas/IRMAS-TrainingData.zip')
        namelist = f.namelist()
        for filename in tqdm(namelist, ascii=True):
            if '.wav' not in filename:
                continue
            wavfile = f.read(filename)
            byt = io.BytesIO(wavfile)
            wavs.append(wav_read(byt)[1].astype('float32'))
            labels.append(filename.split('/')[-2])
    
        for part in ['1', '2', '3']:
            f = zipfile.ZipFile(path+'irmas/IRMAS-TestingData-Part{}.zip'.format(part))
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

