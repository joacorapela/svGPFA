
# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'svGPFA'
copyright = '2019, Lea Duncker and Maneesh Sahani'
author = 'Lea Duncker and Maneesh Sahani'

html_permalinks_icon = '<span>#</span>'
html_theme = 'sphinxawesome_theme'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinxcontrib.bibtex',
    'sphinx_gallery.gen_gallery',
    'sphinxcontrib.proof',
    'sphinx.ext.viewcode',
    ]

bibtex_bibfiles = ['latentsVariablesModels.bib','stats.bib','machineLearning.bib','informationTheory.bib']
bibtex_default_style = 'alpha'

sphinx_gallery_conf = {
    'examples_dirs': '../../../examples/sphinx_gallery/',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'ignore_pattern': r'iblUtils\.py',
}

# for numbering theorems
numfig = True
