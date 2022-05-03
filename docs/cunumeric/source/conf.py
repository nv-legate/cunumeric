# Copyright 2021-2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

sys.path.insert(0, os.path.abspath("comparison"))
sys.path.insert(0, os.path.abspath("../../../"))
import _comparison_generator  # noqa: E402

# Generate comparison table.
with open("comparison/comparison_table.rst.inc", "w") as f:
    f.write(_comparison_generator.generate("cunumeric"))

# -- Project information -----------------------------------------------------

project = "cunumeric"
copyright = "2021-2022, NVIDIA"
author = "NVIDIA"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_markdown_tables",
    "recommonmark",
    "cunumeric.sphinxext.ufunc_formatter",
]

copybutton_prompt_text = ">>> "

# The master toctree document.
master_doc = "index"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

pygments_style = "sphinx"

nitpick_ignore = [
    ("py:class", "type"),
    ("py:class", "scalar"),
    ("py:class", "array_like"),
    ("py:class", "optional"),
    ("py:class", "data-type"),
    ("py:class", "M"),
    ("py:class", "N"),
    ("py:class", "nested list of array_like"),
    ("py:class", "scalars"),
    ("py:class", "complex ndarray"),
    ("py:class", "sequence of ints"),
    ("py:class", "array"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Config napolean
napoleon_custom_sections = [("Availability", "returns_style")]

autosummary_generate = True

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"


def setup(app):
    app.add_js_file("copybutton_pydocs.js")
    app.add_css_file("params.css")
