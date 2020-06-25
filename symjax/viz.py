#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Randall Balestriero"

try:
    import pygraphviz as pgv
except ImportError:
    raise ImportError("Missing pygraphviz dependency.")

from . import tensor as T


def compute_graph(var, graph=None):
    if graph is None:
        graph = G = pgv.AGraph(strict=False, directed=True)
    graph.add_node(str(var))
    node = graph.get_node(var)
    node.attr["style"] = "filled"
    if type(var) == T.Op:
        node.attr["color"] = "green"
    elif type(var) == T.Variable:
        node.attr["shape"] = "rectangle"
        node.attr["color"] = "yellow"
    elif type(var) == T.Placeholder:
        node.attr["shape"] = "rectangle"
        node.attr["color"] = "blue"
    else:
        node.attr["shape"] = "rectangle"
        node.attr["color"] = "grey"

    if type(var) == T.Op:
        for v in var.args:
            try:
                graph.get_node(str(v))
            except KeyError:
                graph = compute_graph(v, graph)
                graph.add_edge(str(v), str(var))
        for v in var.kwargs:
            try:
                graph.get_node(str(var.kwargs[v]))
            except KeyError:
                graph = compute_graph(var.kwargs[v], graph)
                graph.add_edge(str(var.kwargs[v]), str(var))
    return graph
