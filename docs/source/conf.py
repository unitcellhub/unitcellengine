import os
import sys
import sphinx_readable_theme

sys.path.insert(0, os.path.abspath("../../unitcell"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "UnitcellEngine"
copyright = "2024, Ryan Watkins"
author = "Ryan Watkins"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
]

autosummary_generate = True
numpydoc_class_members_toctree = True
numpydoc_show_class_members = False

templates_path = ["_templates"]
exclude_patterns = ["build", "_autosummary"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "readable"
html_theme_path = [sphinx_readable_theme.get_html_theme_path()]
html_static_path = ["_static"]
