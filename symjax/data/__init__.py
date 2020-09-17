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
    cub200,
    audiomnist,
    speech_commands,
    picidae,
    birdvox_70k,
    esc,
    gtzan,
    irmas,
    freefield1010,
    sonycust,
    stl10,
    TUTacousticscenes2017,
    groove_MIDI,
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
