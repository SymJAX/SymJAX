#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__      = "Randall Balestriero"


import symjax
import numpy

def test_add():
    a = symjax.tensor.ones(2)
    assert symjax.tensor.get(a.max()) == 1

test_add()
