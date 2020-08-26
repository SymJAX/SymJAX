from tqdm import tqdm
import matplotlib.image as mpimg
import zipfile
import numpy as np
import os
import requests
import time
import urllib.request
from urllib.request import urlretrieve


def download_file_from_google_drive(file_id, dest, md5=None):
    """Download a Google Drive file from  and place it in root.
    Args:
        file_id (str): id of file to be downloaded
        dist (str): name of file to be saved
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        :param dest:
    """
    # Based on https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    import requests

    url = "https://docs.google.com/uc?export=download"

    if os.path.isfile(dest) and check_integrity(dest, md5):
        print("Using downloaded and verified file: " + fpath)
    else:
        session = requests.Session()

        response = session.get(url, params={"id": file_id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {"id": file_id, "confirm": token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, dest)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response, destination, chunk_size=32768):
    with open(destination, "wb") as f:
        pbar = tqdm(total=None)
        progress = 0
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                progress += len(chunk)
                pbar.update(progress - pbar.n)
        pbar.close()


class DLProgress(tqdm):
    """
    Handle Progress Bar while Downloading
    """

    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        """
        A hook function that will be called once on establishment of the network connection and
        once after each block read thereafter.
        :param block_num: A count of blocks transferred so far
        :param block_size: Block size in bytes
        :param total_size: The total size of the file. This may be -1 on older FTP servers which do not return
                            a file size in response to a retrieval request.
        """
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_extract(path):
    """
    Download and extract database
    :param path:
    """
    url = "https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip"
    hash_code = "00d2c5bc6d35e252742224ab0c1e8fcb"
    save_path = os.path.join(path, "celeba.zip")

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(save_path):
        with DLProgress(
            unit="B", unit_scale=True, miniters=1, desc="Downloading celebA"
        ) as pbar:
            urlretrieve(url, save_path, pbar.hook)


#    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
#        '{} file is corrupted.  Remove the file and try again.'.format(save_path)


def read_image_from_archive(arhive, image_name):
    imgfile = archive.open(image_name, "r")
    img = mpimg.imread(imgfile)
    return img


class celebA:
    """Celebrity faces."""

    file_list = [
        # File ID                         MD5 Hash                            Filename
        (
            "0B7EVK8r0v71pZjFTYXZWM3FlRnM",
            "00d2c5bc6d35e252742224ab0c1e8fcb",
            "img_align_celeba.zip",
        ),
        # ("0B7EVK8r0v71pbWNEUjJKdDQ3dGc", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_align_celeba_png.7z"),
        # ("0B7EVK8r0v71peklHb0pGdDl6R28", "b6cd7e93bc7a96c2dc33f819aa3ac651", "img_celeba.7z"),
        (
            "0B7EVK8r0v71pblRyaVFSWGxPY0U",
            "75e246fa4810816ffd6ee81facbd244c",
            "list_attr_celeba.txt",
        ),
        (
            "1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
            "32bd1bd63d3c78cd57e08160ec5ed1e2",
            "identity_CelebA.txt",
        ),
        (
            "0B7EVK8r0v71pbThiMVRxWXZ4dU0",
            "00566efa6fedff7a56946cd1c10f1c16",
            "list_bbox_celeba.txt",
        ),
        (
            "0B7EVK8r0v71pd0FJY3Blby1HUTQ",
            "cc24ecafdb5b50baae59b03474781f8c",
            "list_landmarks_align_celeba.txt",
        ),
        # ("0B7EVK8r0v71pTzJIdlJWdHczRlU", "063ee6ddb681f96bc9ca28c6febb9d1a", "list_landmarks_celeba.txt"),
        (
            "0B7EVK8r0v71pY0NSMzRuSXJEVkk",
            "d32c9cbf5e040fd4025c592c306e6668",
            "list_eval_partition.txt",
        ),
    ]

    @staticmethod
    def download(path):
        """
        Download the face dataset and store the result into the given
        path

        Parameters
        ----------

            path: str
                the path where the downloaded files will be stored. If the
                directory does not exist, it is created.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        for file in celebA.file_list:
            save_path = os.path.join(path, file[-1])
            if not os.path.exists(save_path):
                download_file_from_google_drive(file[0], save_path)

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

        celebA.download(path + "celebA/")

        t0 = time.time()

        archive = zipfile.ZipFile(path + "celebA/img_align_celeba.zip", "r")
        names = archive.namelist()
        print([name for name in names if name[-1] == "t"])
        names = [name for name in names if name[-3:] == "jpg"]
        names = names[:3000]
        data = np.empty((len(names), 218, 178, 3))
        attributes = np.loadtxt(path + "list_attr_celeba.txt")
        for i, name in enumerate(names):
            data[i] = read_image_from_archive(archive, name)

        print("Dataset celebA loaded in {0:.2f}s.".format(time.time() - t0))

        return data, attributes
