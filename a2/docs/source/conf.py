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

#from pallets_sphinx_themes import ProjectLink
import pytorch_sphinx_theme

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "Int Sys Assignment 2"
copyright = "2020, Magd Bayoumi, Samar Khanna"
author = "Magd Bayoumi, Samar Khanna"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "pallets_sphinx_themes",
    "sphinx.ext.todo",
    'sphinx.ext.autosummary',
    # 'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinxcontrib.katex',
]

# autosectionlabel throws warnings if section names are duplicated.
# The following tells autosectionlabel to not throw a warning for
# duplicated section names that are in different documents.
autosectionlabel_prefix_document = True


# katex options
# katex_options = r'''macros: {
#     "\\"
# }
# '''

katex_prerender = False
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = ".rst"
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Disable docstring inheritance
autodoc_inherit_docstrings = False


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = "sphinx_rtd_theme"

# html_theme = "flask"
# html_theme_options = {"index_sidebar_logo": True}
# html_context = {
#     "project_links": [
#         ProjectLink(
#             "Source Code",
#             "https://dev.azure.com/cornelldatascience/SnapBee/_git/pyfaktory",
#         ),
#         ProjectLink("CDS Team", "https://cornelldata.science"),
#     ]
# }
# html_sidebars = {
#     "index": ["project.html", "localtoc.html", "searchbox.html"],
#     "**": ["localtoc.html", "relations.html", "searchbox.html"],
# }

html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    'pytorch_project': 'docs',
    'canonical_url': 'https://google.com/',#'https://pytorch.org/docs/stable/',
    'collapse_navigation': False,
    'display_version': True,
    'logo_only': True,
}

html_logo = '_static/logo_a1.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    'css/jit.css',
]


# Called automatically by Sphinx, making this `conf.py` an "extension".
def setup(app):
    # NOTE: in Sphinx 1.8+ `html_css_files` is an official configuration value
    # and can be moved outside of this function (and the setup(app) function
    # can be deleted).
    html_css_files = [
        'https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css'
    ]

    # In Sphinx 1.8 it was renamed to `add_css_file`, 1.7 and prior it is
    # `add_stylesheet` (deprecated in 1.8).
    add_css = getattr(app, 'add_css_file', app.add_stylesheet)
    for css_file in html_css_files:
        add_css(css_file)

# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'A2doc'

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'A2.tex', 'A2 Documentation',
     'A2 Contributors', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'A2', 'A2 Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'A2', 'A2 Documentation',
     author, 'A2', 'One line description of project.',
     'Miscellaneous'),
]


# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
}

