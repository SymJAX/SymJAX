#!/usr/bin/env python
# -*- coding: utf-8 -*-
# flake8: noqa


__all__ = ["data",
           "tensor",
           "losses",
           "initializers",
           "layers",
           "optimizers",
           "schedules"]

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
    get_ops)

_current_scope = '/'
_current_graph = [Graph('')]
_variables = {}
_placeholders = {}
_updates = {}
_ops = {}
