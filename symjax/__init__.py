#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

from .base import (
        current_graph,
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


from . import data
from . import tensor
from . import losses
from . import initializers
from . import layers
from . import optimizers
from . import schedules
from . import viz


_current_scope = '/'
_current_graph = [Graph('')]
_variables = {}
_placeholders = {}
_updates = {}
_ops = {}
