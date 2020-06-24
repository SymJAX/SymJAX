#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"


import symjax
import numpy as np
import symjax.tensor as tt
import jax
import jax.numpy as jnp
import jax.scipy as jsp


def test_base():
    a = symjax.tensor.random.randn(())
    print(a in symjax.current_graph())
    f = symjax.function(outputs=a)
    print([f() for i in range(100)])


if __name__ == '__main__':
    test_base()
