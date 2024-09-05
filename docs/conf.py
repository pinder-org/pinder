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
import os
import sys
from pathlib import Path

import pinder

DOC_PATH = Path(__file__).parent

# Avoid verbose logs in rendered notebooks
os.environ["PINDER_LOG_LEVEL"] = "0"

# Include documentation in PYTHONPATH
# in order to import modules for API doc generation etc.
sys.path.insert(0, str(DOC_PATH))
import viewcode

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
    "jupyter_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    # "sphinxcontrib.bibtex",
    "numpydoc",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.githubpages",
    "sphinx_design",
    "sphinx_copybutton",
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
autosummary_generate = True

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

linkcode_resolve = viewcode.linkcode_resolve

project = "PINDER"
copyright = "2024, PINDER Development Team"
author = "PINDER Development Team"
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.

#### HTML ####
html_theme = "pydata_sphinx_theme"


html_static_path = ["_static"]
html_logo = "_static/pinder.png"
html_css_files = [
    "custom.css",
    # Get fonts from Google Fonts CDN
    "https://fonts.googleapis.com/css2"
    "?family=Geologica:wght@100..900"
    "&family=Montserrat:ital,wght@0,100..900;1,100..900"
    "&display=swap",
]
html_js_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/require.js/2.3.4/require.min.js"
]

html_favicon = "_static/favicon.ico"
htmlhelp_basename = "PinderDoc"
html_baseurl = "https://pinder-org.github.io/pinder/"

html_theme_options = {
    "header_links_before_dropdown": 7,
    "pygments_light_style": "friendly",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/pinder-org/pinder/",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
        {
            "name": "Article",
            "url": "https://www.biorxiv.org/content/10.1101/2024.07.17.603980v4",
            "icon": "fa-solid fa-file-lines",
            "type": "fontawesome",
        },
    ],
    "use_edit_page_button": True,
    "show_prev_next": False,
    "show_toc_level": 2,
}
html_sidebars = {
    # No primary sidebar for these pages
    "readme": [],
    "example_readme": [],
    "faq": [],
    "limitations": [],
    "changelog": [],
}
html_context = {
    "github_user": "pinder-org",
    "github_repo": "pinder",
    "github_version": "main",
    "doc_path": "doc",
}
html_scaled_image_link = False
