#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
"""
CIFAR10 Dataset
===============


This example shows how to download/load/import CIFAR10
"""


import symjax
import matplotlib.pyplot as plt

cifar10 = symjax.data.cifar10.load()

plt.figure(figsize=(10, 4))
for i in range(10):

    plt.subplot(2, 5, 1 + i)

    image = cifar10['train_set/images'][i]
    label = cifar10['train_set/labels'][i]

    plt.imshow(image.transpose((1, 2, 0)) / image.max(), aspect='auto',
               cmap='Greys')
    plt.xticks([])
    plt.yticks([])
    plt.title('{}:{}'.format(label, cifar10['label_to_name'][label]))

plt.tight_layout()
