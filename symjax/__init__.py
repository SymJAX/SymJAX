#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

from .base import *

__all__ = [
    "datasets",
    "tensor",
    "losses",
    "interpolation",
    "utils",
    "initializers",
    "layers",
    "schedules",
    "optimizers"]

from . import *

_current_scope = '/'
_current_graph = [Graph('')]
_variables = {}
_placeholders = {}
_updates = {}
_ops = {}
