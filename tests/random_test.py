#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"


import symjax


def test_base():
    a = symjax.tensor.random.randn(())
    print(a in symjax.current_graph())
    f = symjax.function(outputs=a)
    print([f() for i in range(100)])


if __name__ == "__main__":
    test_base()
