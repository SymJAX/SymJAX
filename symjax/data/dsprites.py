import numpy as np
from .utils import download_dataset


# https://github.com/deepmind/dsprites-dataset

_urls = {
    "https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz": "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
}


def load(path=None):
    """greyscale image classification and disentanglement

    This dataset consists of 737,280 images of 2D shapes, procedurally generated
    from 5 ground truth independent latent factors, controlling the shape, scale,
    rotation and position of a sprite. This data can be used to assess the
    disentanglement properties of unsupervised learning methods.

    dSprites is a dataset of 2D shapes procedurally generated from 6 ground
    truth independent latent factors. These factors are color, shape, scale,
    rotation, x and y positions of a sprite.

    All possible combinations of these latents are present exactly once,
    generating N = 737280 total images.

    https://github.com/deepmind/dsprites-dataset

        Parameters
        ----------
            path: str (optional)
                default ($DATASET_PATH), the path to look for the data and
                where the data will be downloaded if not present

        Returns
        -------

            images: array

            latent: array

            classes: array

    """

    if path is None:
        path = os.environ["DATASET_PATH"]

    download_dataset(path, _dataset, _urls)

    dataset_zip = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
    imgs = dataset_zip["imgs"]
    latents_values = dataset_zip["latents_values"]
    latents_classes = dataset_zip["latents_classes"]
    metadata = dataset_zip["metadata"][()]
    data = {
        "train_set/images": imgs,
        "train_set/latent_values": latents_values,
        "train_set/latents_classes": latents_classes,
        "metadata": metadata,
    }
    return data
