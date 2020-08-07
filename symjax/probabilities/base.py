#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

from .. import tensor as T
from .. import nn
import numpy as np


class Categorical:
    def __init__(self, probabilities=None, n=1, logits=None):
        self.n = n
        if probabilities is not None:
            self.logits = T.log(probabilities)
            self.probabilities = probabilities
        else:
            self.probabilities = nn.softmax(logits)
            self.logits = logits

    def log_prob(self, value):
        return nn.log_softmax(T.take(self.logits, value))

    def prob(self, value):
        return T.exp(self.log_prob(value))

    def sample(self):
        return T.random.categorical(self.logits, shape=(self.n,))

    def entropy(self):
        return -T.sum(self.probabilities * nn.log_softmax(self.logits))
