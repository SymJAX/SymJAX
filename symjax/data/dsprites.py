import numpy as np


# https://github.com/deepmind/dsprites-dataset
class dsprites:
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
    """

    def download(path):

        """
        Download the MNIST dataset and store the result into the given
        path

        Parameters
        ----------

            path: str
                the path where the downloaded files will be stored. If the
                directory does not exist, it is created.
        """

        # Check if directory exists
        if not os.path.isdir(path + "dsprites"):
            print("Creating mnist Directory")
            os.mkdir(path + "dsprites")

        filename = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
        # Check if file exists
        if not os.path.exists(path + "dsprites/" + filename):
            td = time.time()
            url = "https://github.com/deepmind/dsprites-dataset/blob/master/" + filename
            urllib.request.urlretrieve(url, path + "dsprites/filename.npz")

    def load(path=None):
        """
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

        download(path)

        t0 = time.time()
        dataset_zip = np.load("dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")
        imgs = dataset_zip["imgs"]
        latents_values = dataset_zip["latents_values"]
        latents_classes = dataset_zip["latents_classes"]
        metadata = dataset_zip["metadata"][()]
        return imgs, latent_values, latents_classes
