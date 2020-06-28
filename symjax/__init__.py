#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from . import data
from . import tensor
from . import nn
from . import probabilities
from .base import *


_graphs = [Graph("")]

__all__ = ["data", "tensor", "nn", "probabilities"]
