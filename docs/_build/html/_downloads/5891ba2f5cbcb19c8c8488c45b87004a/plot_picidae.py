#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa
"""
Speech picidae Dataset
=======================


This example shows how to download/load/import speech picidae
"""


import symjax
import matplotlib.pyplot as plt

picidae = symjax.data.picidae.load()

plt.figure(figsize=(10, 4))
for i in range(10):
    
    plt.subplot(2, 5, 1 + i)
    plt.plot(picidae['wavs'][i])
    plt.title(str(picidae['labels'][i]))

plt.tight_layout()
