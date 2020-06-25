# Based on a script by Seiya Tokui. With the following copyright
# Copyright (c) 2014 Seiya Tokui
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# Given the wnid of a synset, the wnid of hyponym synsets can be obtained at
# http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=[wnid]
#
# To obtain the full hyponym (the synset of the whole subtree starting
# from wnid), you can request
# http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid=[wnid]&full=1
#
# to get the word of s synset
# http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=[wnid]
#
# Given the wnid of a synset, the URLs of its images can be obtained at
# http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=[wnid]
#
# mappingfrom all synset to words
# http://image-net.org/archive/words.txt
#


import argparse
import urllib.request, urllib.error, urllib.parse
import time
import os
import math
import threading
import sys
import imghdr
import http.client
from ssl import CertificateError


class DownloadError(Exception):
    """Base class for exceptions in this module."""

    def __init__(self, message=""):
        self.message = message


def download(n_images, min_size, n_threads, wnids_list, out_dir):
    wnid_thread_lists = list()
    wnid_list_len = len(wnids_list)
    wnid_thread_sizes = int(math.ceil(float(wnid_list_len) / n_threads))
    for i in range(n_threads):
        wnid_thread_lists.append(
            wnids_list[i * wnid_thread_sizes : (i + 1) * wnid_thread_sizes]
        )

    # Define the threads
    def downloader(wnid_list):
        for wnid in wnid_list:
            dir_name = wnid
            print("Downloading " + dir_name)
            dir_path = os.path.join(out_dir, dir_name)
            if os.path.isdir(dir_path):
                print("skipping: already have " + dir_name)
            else:
                image_url_list = get_image_urls(wnid)
                download_images(dir_path, image_url_list, n_images, min_size)

    # initialize the threads
    print(wnid_thread_lists[0])
    download_threads = [
        threading.Thread(target=downloader, args=([wnid_thread_lists[i]]))
        for i in range(n_threads)
    ]

    for t in download_threads:
        t.start()

    is_alive = True
    while is_alive:
        is_alive = False
        for t in download_threads:
            is_alive = is_alive or t.isAlive()
        time.sleep(0.1)

    for t in download_threads:
        t.join()
    print("finished")


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_url_request_list_function(request_url):
    def get_url_request_list(wnid, timeout=5, retry=3):
        url = request_url + wnid
        f = urllib.request.urlopen(url)
        response = f.read().decode()
        f.close()
        print("response: " + response)
        list = str.split(response)
        return list

    return get_url_request_list


get_image_urls = get_url_request_list_function(
    "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
)

get_subtree_wnid = get_url_request_list_function(
    "http://www.image-net.org/api/text/wordnet.structure.hyponym?wnid="
)

get_full_subtree_wnid = get_url_request_list_function(
    "http://www.image-net.org/api/text/wordnet.structure.hyponym?full=1&wnid="
)


def get_words_wnid(wnid):
    url = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid=" + wnid
    f = urllib.request.urlopen(url)
    content = f.read().decode()
    f.close()
    return content


def download_images(dir_path, image_url_list, n_images, min_size):
    mkdir(dir_path)
    image_count = 0
    for url in image_url_list:
        if image_count == n_images:
            break
        try:
            f = urllib.request.urlopen(url)
            image = f.read()
            f.close()
            extension = imghdr.what("", image)  # check if valid image
            if extension == "jpeg":
                extension = "jpg"
            if sys.getsizeof(image) > min_size:
                image_name = "image_" + str(image_count) + "." + extension
                image_path = os.path.join(dir_path, image_name)
                image_file = open(image_path, "wb")
                image_file.write(image)
                image_file.close()
                image_count += 1
        except:
            print("skipping ", url)


def main(wnid, out_dir, n_threads, n_images, fullsubtree, noroot, nosubtree, min_size):

    wnids_list = []

    # First get the list of wnids
    if not noroot:
        wnids_list.append(wnid)
    if not nosubtree:
        if fullsubtree:
            subtree = get_full_subtree_wnid(wnid)
        else:
            subtree = get_subtree_wnid(wnid, timeout, retry)
        for i in range(1, len(subtree)):
            subtree[i] = subtree[i][1:]  # removes dash
        wnids_list.extend(subtree)

    # create root directory
    mkdir(out_dir)
    download(n_images, min_size, n_threads, wnids_list, out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("wnid", help="Imagenet wnid, example n03489162")
    p.add_argument("outdir", help="Output directory")
    p.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="Number of parallel threads to download",
    )
    p.add_argument(
        "--images",
        "-i",
        type=int,
        default=20,
        metavar="N_IMAGES",
        help="Number of images per category to download",
    )
    p.add_argument(
        "--fullsubtree", "-F", action="store_true", help="Downloads the full subtree"
    )
    p.add_argument(
        "--noroot", "-R", action="store_true", help="Do not Downloads the root"
    )
    p.add_argument(
        "--nosubtree", "-S", action="store_true", help="Do not Downloads the subtree"
    )

    p.add_argument(
        "--humanreadable",
        "-H",
        action="store_true",
        help="Makes the folders human readable",
    )

    p.add_argument(
        "--minsize",
        "-m",
        type=float,
        default=7000,
        help="Min size of the images in bytes",
    )

    args = p.parse_args()
    main(
        wnid=args.wnid,
        out_dir=args.outdir,
        n_threads=args.jobs,
        n_images=args.images,
        fullsubtree=args.fullsubtree,
        noroot=args.noroot,
        nosubtree=args.nosubtree,
        min_size=args.minsize,
    )
