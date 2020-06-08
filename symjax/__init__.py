#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

from .base import (
        Graph,
        function,
        gradients,
        jacobians,
        reset,
        save,
        load,
        variable,
        placeholder,
        op)


from . import datasets
from . import tensor
from . import losses
from . import utils
from . import initializers
from . import layers
from . import optimizers
from . import schedules


_current_scope = '/'
_current_graph = [Graph('')]
_variables = {}
_placeholders = {}
_updates = {}
_ops = {}
