#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

import symjax.tensor as T
import matplotlib.pyplot as plt
from symjax.viz import compute_graph

x = T.random.randn((10,), name="x")
y = T.random.randn((10,), name="y")
z = T.random.randn((10,), name="z")

w = T.Variable(T.ones(1), name="w")
out = (x + y).sum() * w + z.sum()

graph = compute_graph(out)
graph.draw("file.png", prog="dot")


import matplotlib.image as mpimg

img = mpimg.imread("file.png")
plt.figure(figsize=(15, 5))
imgplot = plt.imshow(img)
plt.xticks()
plt.yticks()
plt.tight_layout()
