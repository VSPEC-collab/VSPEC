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
sys.path.insert(0, os.path.abspath('../../VSPEC'))


# -- Project information -----------------------------------------------------

project = 'VSPEC'
copyright = '2023, The VSPEC Collaboration'
author = 'Ted Johnson and Cameron Kelahan'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx_automodapi.automodapi',
    'sphinx_automodapi.smart_resolver',
    'numpydoc',
    'sphinx.ext.intersphinx',
    'nbsphinx',
    'nbsphinx_link',
    'sphinxcontrib.bibtex',
    'sphinx_gallery.gen_gallery',
    # 'sphinx.ext.autosummary',
    # 'sphinx.ext.autodoc'
]
numpydoc_show_class_members = False
bibtex_bibfiles = ['refs.bib']

# autosummary_generate = True
# autodoc_default_options = {
#     'template': 'custom_template.rst',
#     # other options...
# }


# Settings for sphinx gallery
sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'matplotlib_animations': True,
}



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
html_theme = 'furo'

# html_theme_options = {
#     "light_css_variables": {
#         "color-brand-primary": "red",
#         "color-brand-content": "#CC3333",
#         "color-admonition-background": "orange",
#     },
# }


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/table_fix.css',
    'css/custom.css'
]

# Logo and favicon
html_logo = '_static/images/vspec_logo.png'
html_favicon = '_static/images/favicon.ico'

# Configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/',
               (None, 'http://data.astropy.org/intersphinx/python3.inv')),
    'numpy': ('https://numpy.org/doc/stable/',
              (None, 'http://data.astropy.org/intersphinx/numpy.inv')),
    'pandas': ('https://pandas.pydata.org/docs/',
              (None, 'http://data.astropy.org/intersphinx/pandas.inv')),
    'scipy': ('https://docs.scipy.org/doc/scipy/',
              (None, 'http://data.astropy.org/intersphinx/scipy.inv')),
    'matplotlib': ('https://matplotlib.org/stable/',
                   (None, 'http://data.astropy.org/intersphinx/matplotlib.inv')),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'h5py': ('https://docs.h5py.org/en/stable/', None),
}


bibtex_reference_style = 'author_year'

bibtex_default_style = 'unsrt'