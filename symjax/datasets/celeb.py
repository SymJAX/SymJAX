import requests
import tqdm
import more_itertools
import matplotlib.image as mpimg
import zipfile
import matplotlib.pyplot as plt
import numpy as np

def download_file_from_google_drive(id, destination):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            # turn the generator into a list
            gen = response.iter_content(CHUNK_SIZE)
            for chunk in tqdm.tqdm(gen, total=44239):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def read_image_from_archive(arhive, image_name):
    imgfile = archive.open(image_name, 'r')
    img = mpimg.imread(imgfile)
    return img


if __name__ == "__main__":
    file_id, destination = '0B7EVK8r0v71pZjFTYXZWM3FlRnM', 'celebA.zip'
    #download_file_from_google_drive(file_id, destination)
    archive = zipfile.ZipFile(destination, 'r')
    names = archive.namelist()
    names = [name for name in names if name[-3:] == 'jpg']
    names = names[:3000]
    data = np.empty((len(names), 218, 178, 3))
    for i, name in enumerate(names):
        data[i] = read_image_from_archive(archive, name)


