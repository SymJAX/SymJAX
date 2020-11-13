#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"


import symjax
import symjax.tensor as T
import numpy as np


def test_base():
    a = T.ones((10,))
    b = a.sum()
    print(b.get())
    print(b.get())
    f = symjax.function(outputs=b)
    [f() for i in range(100)]


if __name__ == "__main__":
    test_base()
