#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

__all__ = ['data',
           'tensor',
           'losses',
           'initializers',
           'layers',
           'optimizers',
           'schedules']

from . import data
from . import tensor
from . import losses
from . import initializers
from . import layers
from . import optimizers
from . import schedules

from .base import (
    current_graph,
    Graph,
    function,
    gradients,
    jacobians,
    reset_variables,
    save_variables,
    load_variables,
    get_variables,
    get_placeholders,
    get_ops,
    get_updates)


_current_scope = '/'
_current_graph = [Graph('')]
_variables = {}
_placeholders = {}
_updates = {}
_ops = {}
