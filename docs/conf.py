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
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import snsim

project = 'snsim'
copyright = '2021, Carreres'
author = 'Carreres'

# The full version, including alpha/beta/rc tags
release = snsim.__version__

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'numba': ('https://numba.readthedocs.io/en/stable/', None),
    'sncosmo': ('https://sncosmo.readthedocs.io/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None)}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = ['myst_parser', 'sphinx.ext.napoleon', 'sphinx_markdown_tables', 'sphinx.ext.autosectionlabel',
              'sphinx.ext.linkcode', 'sphinx.ext.intersphinx']

myst_enable_extensions = ["dollarmath"]
myst_dmath_double_inline = True

autosectionlabel_prefix_document = True
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_logo = "_static/snsimlogo.webp"

html_theme_options = {
                        "sidebar_hide_name": True,
                     }

import inspect
from os.path import relpath


# Copied on snsim documentation
def linkcode_resolve(domain, info):
    """Determine the URL corresponding to Python object."""
    if domain != 'py':
        return None

    modname = info['module']
    fullname = info['fullname']

    submod = sys.modules.get(modname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except:
            return None

    try:
        fn = inspect.getsourcefile(obj)
    except:
        fn = None
    if not fn:
        return None

    try:
        source, lineno = inspect.findsource(obj)
    except:
        lineno = None

    if lineno:
        linespec = "#L%d" % (lineno + 1)
    else:
        linespec = ""

    fn = relpath(fn, start=snsim.__snsim_dir_path__)
    if 'dev' in snsim.__version__:
        return "https://github.com/bcarreres/snsim/tree/dev/snsim/%s%s" % (fn, linespec)
    else:
        return "https://github.com/bcarreres/snsim/tree/main/snsim/%s%s" % (fn, linespec)
