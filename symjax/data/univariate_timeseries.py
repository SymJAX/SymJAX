import os
import pickle, gzip
import urllib.request
import numpy as np
import time
import zipfile
from scipy.io import arff
import io
from tqdm import tqdm


class univariate_timeseries:
    """univariate_timeseries"""

    @staticmethod
    def download(path):
        """
        Download the univariate_timeseries dataset
        path

        Parameters
        ----------

            path: str
                the path where the downloaded files will be stored. If the
                directory does not exist, it is created.
        """

        # Check if directory exists
        if not os.path.isdir(path + "univariate_timeseries"):
            print("Creating univariate_timeseries Directory")
            os.mkdir(path + "univariate_timeseries")

        # Check if file exists
        if not os.path.exists(path + "univariate_timeseries/Univariate2018_arff.zip"):
            td = time.time()
            print("Creating univariate timeseries")
            url = "http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_arff.zip"
            urllib.request.urlretrieve(
                url, path + "univariate_timeseries/Univariate2018_arff.zip"
            )

    @staticmethod
    def load(path=None):
        """
        Parameters
        ----------
            path: str (optional)
                default ($DATASET_PATH), the path to look for the data and
                where the data will be downloaded if not present

        Returns
        -------

            train_images: array

            train_labels: array

            valid_images: array

            valid_labels: array

            test_images: array

            test_labels: array

        """

        if path is None:
            path = os.environ["DATASET_PATH"]

        univariate_timeseries.download(path)

        t0 = time.time()

        # Loading the fileunivariate_timeseries
        f = zipfile.ZipFile(path + "univariate_timeseries/Univariate2018_arff.zip")
        # init. the data array
        all_data = {}
        for name in tqdm(f.namelist()):
            if "TRAIN.txt" in name or "TEST.txt" in name:
                file = f.read(name)
                if "PLAID" in name:
                    data = np.loadtxt(io.BytesIO(file), delimiter=",")
                else:
                    data = np.loadtxt(io.BytesIO(file))

                dataset = name[:-4].split("/")[-2]
                part = "TRAIN" if "TRAIN" in name else "TEST"
                all_data[dataset + "/" + part] = data
        return all_data
