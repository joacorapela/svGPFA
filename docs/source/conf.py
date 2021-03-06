# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))
# sys.path.insert(0, '/nfs/nhome/live/rapela/dev/research/gatsby-swc/gatsby/svGPFA/master/src/')


# -- Project information -----------------------------------------------------

project = 'svGPFA'
copyright = '2019, Lea Duncker and Maneesh Sahani'
author = 'Lea Duncker and Maneesh Sahani'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.intersphinx', 'sphinx.ext.todo', 'sphinx.ext.ifconfig', 'sphinx.ext.viewcode', 'sphinx.ext.inheritance_diagram', 'sphinx.ext.autosummary']
# extensions = ['sphinx.ext.autodoc', 'sphinx.ext.inheritance_diagram', 'autoapi.extension']

# autoapi_type = 'python'
# autoapi_dirs = ['../../src/']

# Added this line because when building the project in readthedocs I get
# an error contents.rst not found
master_doc = 'index'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_static_path = []

# Added by Joaco to solve the memory problem with torch
autodoc_mock_imports = ["torch"]
