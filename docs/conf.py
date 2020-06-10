#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
# Path setup
# ----------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.setrecursionlimit(1500)
import mock

from sphinx_gallery.scrapers import matplotlib_scraper

MOCK_MODULES = ['soundfile']
for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = mock.Mock()


# Project information
# -------------------

project = 'symjax'
copyright = '2020, Randall Balestriero'
author = 'Randall Balestriero'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# General configuration
# ---------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx_rtd_theme',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.coverage',
    'sphinx.ext.autosummary',
    "sphinx_gallery.gen_gallery"]

# matplotlib svg rendering
class MatplotlibSVG(object):
    """Render images with SVG format."""

    def __repr__(self):
        """Let matplotlib know the classname."""
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        """Return image with SVG format."""
        return matplotlib_scraper(*args, format="svg", **kwargs)

# gallery options
sphinx_gallery_conf = {
    "examples_dirs": "../gallery",
    "gallery_dirs": "auto_examples",
    "image_scrapers": (MatplotlibSVG(),),
}

# See https://github.com/rtfd/readthedocs.org/issues/283
mathjax_path = (
    'https://cdn.mathjax.org/mathjax/latest/MathJax.js?'
    'config=TeX-AMS-MML_HTMLorMML')

# see http://stackoverflow.com/q/12206334/562769
numpydoc_show_class_members = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# Options for HTML output
# -----------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# removed _static
html_static_path = []

html_logo = 'img/symjax_logo.png'
html_theme_options = {'logo_only': True}
html_css_files = ['custom.css']
