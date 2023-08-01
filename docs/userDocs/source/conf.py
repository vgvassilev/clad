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
import os

project = "Clad"
copyright = "2014, Vassil Vassilev"
author = "Vassil Vassilev"

# The full version, including alpha/beta/rc tags
release = "2014"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.todo", "sphinx.ext.mathjax"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "github_user": "vgvassilev",
    "github_repo": "clad",
    "github_banner": True,
    "fixed_sidebar": True,
}

highlight_language = "C++"


todo_include_todos = True

current_file_dir = os.path.dirname(os.path.realpath(__file__))
CLAD_ROOT = current_file_dir + "/../../.."

with open(CLAD_ROOT + "/VERSION", "r") as f:
    version = f.read()

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
# Add latex physics package
mathjax3_config = {
    "loader": {"load": ["[tex]/physics"]},
    "tex": {"packages": {"[+]": ["physics"]}},
}
if os.environ.get("CLAD_BUILD_INTERNAL_DOCS"):
    html_extra_path = [CLAD_ROOT + "/build/docs/"]

    import subprocess

    CMAKE_CONFIGURE_COMMAND = (
        "mkdir {0}/build; cd {0}/build; cmake ../ "
        "-DClang_DIR=/usr/lib/llvm-14 -DLLVM_DIR="
        "/usr/lib/llvm-14 -DCLAD_ENABLE_DOXYGEN=ON "
        "-DCLAD_INCLUDE_DOCS=ON"
    ).format(CLAD_ROOT)
    subprocess.call(CMAKE_CONFIGURE_COMMAND, shell=True)

    INTERNAL_DOCS_DIR = "{0}/build/docs/internalDocs".format(CLAD_ROOT)
    RUN_DOXYGEN_COMMAND = (
        "(cat doxygen.cfg; echo 'OUTPUT_DIRECTORY = .') | doxygen -"
    ).format(INTERNAL_DOCS_DIR)
    print(RUN_DOXYGEN_COMMAND)
    subprocess.call(RUN_DOXYGEN_COMMAND, shell=True, cwd=INTERNAL_DOCS_DIR)
