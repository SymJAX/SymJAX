#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from . import ops
from . import Tensor

class Layer(Tensor):
    def __init__(self, internal_ops):
        super().__init__(internal_ops[-1])
        self.internal_ops = internal_ops
        self._input = internal_ops[0].input

    def deter_dict(self, value):
        deter_dict = dict()
        for op in self.internal_ops:
            deter_dict.update(op.deter_dict(value))
        return deter_dict

    def __getitem__(self, key):
        return self.internal_ops[key]

    @property
    def reset_variables_op(self):
        return tf.group(*[op.reset_variables_op for op in self.internal_ops])

    def backward(self, input):
        return tf.gradients(self, self.input, input)[0]

    @property
    def input(self):
        return self._input

    def variables(self, trainable=True):
        variables = list()
        for op in self.internal_ops:
            variables += op.variables(trainable=trainable)
        return variables

    @property
    def updates(self):
        updates = list()
        for op in self.internal_ops:
            updates.append(op.updates)
        return updates

    def __len__(self):
        return len(self.internal_ops)

def Dense(input, filters, nonlinearity=0, W_func=tf.identity):
    dense = ops.Dense(input, filters, b=None, W_func=tf.identity)
    bn = ops.BatchNorm(dense, [0])
    nonlinearity = ops.Activation(bn, nonlinearity)
    return Layer([dense, bn, nonlinearity])

def Conv2D(input, filters, nonlinearity=0, strides=1, pad='valid',
           W_bn=tf.ones, b_bn=tf.zeros, W_func=tf.identity):
    conv = ops.Conv2D(input, filters=filters, strides=strides, b=None, pad=pad,
                      W_func=tf.identity)
    bn = ops.BatchNorm(conv, [0, 2, 3], W=W_bn, b=b_bn)
    nonlinearity = ops.Activation(bn, nonlinearity)
    return Layer([conv, bn, nonlinearity])

def Conv2DPool(input, filters, nonlinearity=0, pad='valid', pool_shape=(2, 2),
               strides=1, W_bn=tf.ones, b_bn=tf.zeros, W_func=tf.identity):
    conv = ops.Conv2D(input, filters=filters, strides=strides, b=None, pad=pad,
                      W_func=W_func)
    bn = ops.BatchNorm(conv, [0, 2, 3], W=W_bn, b=b_bn)
    nonlinearity = ops.Activation(bn, nonlinearity)
    pool = ops.Pool2D(nonlinearity, pool_shape)
    return Layer([conv, bn, nonlinearity, pool])

def ResBlockV1(input, filters, stride=1):
    if stride > 1:
        conv_linear = ops.Conv2D(input, filters=(filters, 3, 3),
                                 strides=stride, b=None, pad='same')
    else:
        conv_linear = ops.Identity(input)
    conv = ops.Conv2D(input, filters=(filters, 3, 3), b=None,
                      pad='same')
    bn = ops.BatchNorm(conv, [0, 2, 3])
    nonlinearity = ops.Activation(bn, 0.)
    if stride > 1:
        pool = ops.Pool2D(nonlinearity, stride)
    else:
        pool = ops.Identity(nonlinearity)
    out_conv = ops.Conv2D(pool, filters=(filters, 3, 3), b=None,
                          pad='same')
    bn2 = ops.BatchNorm(out_conv, [0, 2, 3])
    merge = ops.Merge([bn2, conv_linear], tf.add_n)
    output = ops.Activation(merge, 0.)
    return Layer([conv_linear, conv, bn, nonlinearity, pool, out_conv, bn2,
                  merge, output])


def ResBlockV2(input, filters, stride=1):
    if stride > 1:
        conv_linear = ops.Conv2D(input, filters=(filters, 3, 3),
                                 b=None, pad='same')
    else:
        conv_linear = ops.Identity(input)
    conv = ops.Conv2D(input, filters=(filters, 3, 3), b=None,
                      pad='same')
    bn = ops.BatchNorm(conv, [0, 2, 3])
    nonlinearity = ops.Activation(bn, 0.)
    out_conv = ops.Conv2D(nonlinearity, filters=(filters, 3, 3), b=None,
                          pad='same')
    bn2 = ops.BatchNorm(out_conv, [0, 2, 3])
    merge = ops.Merge([bn2, conv_linear], tf.add_n)
    return Layer([conv_linear, conv, bn, nonlinearity, out_conv, bn2,
                  merge])
