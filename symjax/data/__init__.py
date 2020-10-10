#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import (
    mnist,
    emnist,
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
    rock_paper_scissors,
    dcase_2019_task4,
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
