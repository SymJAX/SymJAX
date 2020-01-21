# https://github.com/deepmind/dsprites-dataset

def download(path):
    # Check if directory exists
    if not os.path.isdir(path+'dsprites'):
        print('Creating mnist Directory')
        os.mkdir(path+'dsprites')

    filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    # Check if file exists
    if not os.path.exists(path+'dsprites/' + filename):
        td  = time.time()
        url = 'https://github.com/deepmind/dsprites-dataset/blob/master/' + filename
#        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, 
#                                                    desc='DL dataset') as t:
        urllib.request.urlretrieve(url,path + 'dsprites/filename.npz')


def load(PATH=None, classes=range(10)):

    if PATH is None:
        PATH = os.environ['DATASET_PATH']

    download(PATH)

    t0 = time.time()
    dataset_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    metadata = dataset_zip['metadata'][()]
    return imgs, latent_values, latents_classes, metadata

