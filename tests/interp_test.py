#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import symjax


def test_image():
    im1 = np.random.rand(3, 2, 2)
    im2 = np.random.rand(3, 4, 4)
    images = symjax.data.utils.resample_images([im1, im2], (4, 4))
    assert np.array_equal(images[0, :, ::3, ::3], im1)
    assert np.array_equal(images[1], im2)


if __name__ == "__main__":
    test_image()
