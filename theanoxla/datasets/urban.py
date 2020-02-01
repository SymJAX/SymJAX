import os
import pickle,gzip
import urllib.request
import numpy as np
import time
from google_drive_downloader import GoogleDriveDownloader as gdd
import tarfile

#from ..utils import  DownloadProgressBar
#https://zenodo.org/record/3233082
def download(path):
    # Check if directory exists
    if not os.path.isdir(path+'urban'):
        print('Creating mnist Directory')
        os.mkdir(path+'urban')

    # Check if file exists
    if not os.path.exists(path+'urban/urban.pkl.gz'):
        td  = time.time()
        gid = 'oLusMz&sa=D&ust=1580537540216000&usg=AFQjCNGYbw_aR5y9JVC8ecYkvYSqJGLPlQ'
        gdd.download_file_from_google_drive(file_id=gid,
                                    dest_path=path+'urban/urban.tar.gz',
                                    unzip=False)

def load(PATH=None, classes=range(10)):
    """Urban.

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

    # Loading the file
    f = tarfile.open(PATH+'urban/urban.tar.gz', 'rb')
    return f
