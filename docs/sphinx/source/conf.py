
# -- Project information -----------------------------------------------------

project = 'svGPFA'
copyright = '2019, Lea Duncker and Maneesh Sahani'
author = 'Lea Duncker and Maneesh Sahani'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinxcontrib.bibtex',
    'sphinx_gallery.gen_gallery',
    ]

bibtex_bibfiles = ['gaussianProcesses.bib','stats.bib']
bibtex_default_style = 'alpha'

sphinx_gallery_conf = {
    'examples_dirs': '../../../examples/sphinx_gallery/',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
}
