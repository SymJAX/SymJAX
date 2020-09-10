#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import (
    mnist,
    fashionmnist,
    dsprites,
    svhn,
    cifar10,
    cifar100,
    imagenette,
    audiomnist,
    speech_commands,
    picidae,
    birdvox_70k,
    esc,
    gtzan,
    irmas,
)

from .utils import (
    patchify_1d,
    patchify_2d,
    train_test_split,
    batchify,
    resample_images,
    download_dataset,
    extract_file,
)
