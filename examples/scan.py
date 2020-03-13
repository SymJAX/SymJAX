#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "../")  
import symjax as sj
import symjax.tensor as T
import numpy as np

__author__      = "Randall Balestriero"


# example of cumulative sum

def func(carry, x):
    return carry+1, 0

output, _ = T.scan(func, T.zeros(1), T.ones(10), length=10)

f = sj.function(outputs=output)
print(f())
# [10.]

# example of simple RNN

w = T.Placeholder((3, 10), 'float32')
h = T.random.randn((3, 3))
b = T.random.randn((3,))
t_steps = 100
X = T.random.randn((t_steps, 10))

def rnn_cell(carry, x, w):
    output = T.sigmoid(T.matmul(w, x)+T.matmul(carry, h)+b)
    return output, output

last, hidden = T.scan(rnn_cell, T.zeros(3), X, constants=(w,))

g = sj.gradients(hidden.sum(), w)
print(g.get({w:np.ones((3, 10))}))
f = sj.function(w, outputs=hidden)
print(f(np.ones((3, 10))))
#
