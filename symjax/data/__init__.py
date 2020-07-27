#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .mnist import load as mnist
from .svhn import load as svhn
from .cifar10 import load as cifar10
from .cifar100 import load as cifar100
from .imagenette import load as imagenette

from .audiomnist import load as audiomnist
from .speech_commands import load as speech_commands
from .picidae import load as picidae

from .utils import (
    patchify_1d,
    patchify_2d,
    train_test_split,
    batchify,
    resample_images,
    download_dataset,
    extract_file,
)
