#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

__all__ = ['data',
           'tensor',
           'nn',
           'probabilities']


from . import data
from . import tensor
from . import nn
from . import probabilities
from .base import *

_current_scope = '/'
_current_graph = [Graph('')]
_variables = {}
_placeholders = {}
_updates = {}
_ops = {}
