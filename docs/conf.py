# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import sys
from os.path import realpath, dirname, join

doc_path = dirname(realpath(__file__))
package_path = join(dirname(doc_path), "src")

sys.path.insert(0, doc_path)


# -- Project information -----------------------------------------------------

project = "pinder"
copyright = "2023, VantAI"
author = "VantAI"


def get_release():
    """The full version, including alpha/beta/rc tags."""
    import pinder.core as pinder

    return pinder.__version__


release = get_release()
version = get_release()


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    # "sphinx.ext.linkcode",
    # "sphinxcontrib.bibtex",
    "numpydoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "myst_nb",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
    ".md": "myst-nb",
}

nb_custom_formats = {".ipynb": ["jupytext.reads", {"fmt": "ipynb"}]}
nb_execution_timeout = 720
nb_kernel_rgx_aliases = {"pinder.*": "python3"}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates", "templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "build", "Thumbs.db", ".DS_Store"]

master_doc = "index"

pygments_style = "sphinx"

todo_include_todos = False

# Prevents numpydoc from creating an autosummary which does not work
# properly due to import system
numpydoc_show_class_members = False

napoleon_use_param = True
napoleon_use_rtype = True

# Prevent autosummary from using sphinx-autogen, since it would
# overwrite the document structure given by apidoc.json
autosummary_generate = False

autodoc_member_order = "bysource"
autodoc_default_options = {"ignore-module-all": True}

myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

myst_url_schemes = ("http", "https", "mailto")
# nb_execution_mode = "force"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

#### HTML ####
html_theme = "piccolo_theme"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_logo = "_static/pinder.png"
html_theme_options = {
    "source_url": "https://github.com/pinder-org/pinder/",
    "source_icon": "github",
    "show_theme_credit": False,
}

html_css_files = [
    "custom.css",
]
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

html_favicon = "_static/favicon.ico"
htmlhelp_basename = "PinderDoc"
