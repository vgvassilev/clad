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
import datetime
import os

project = "Clad"
# Computed rather than written down, so it does not quietly go stale the way
# the release string did.
copyright = f"2014-{datetime.date.today().year}, Vassil Vassilev"
author = "Vassil Vassilev"

# release is set from the VERSION file further down, once CLAD_ROOT is known.


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ["sphinx.ext.todo", "sphinx.ext.mathjax", "sphinxcontrib.mermaid"]

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
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = ["clad.css"]

# Furo takes a repository rather than alabaster's banner, and uses it to put an
# "Edit this page" link on every page.
html_theme_options = {
    "source_repository": "https://github.com/vgvassilev/clad/",
    "source_branch": "master",
    "source_directory": "docs/userDocs/source/",
}

highlight_language = "C++"

# GitHub's own highlighting, so a snippet reads the same here as it does in the
# repository. github-dark ships with Pygments; github-light comes from
# accessible-pygments, which furo depends on.
pygments_style = "github-light"
pygments_dark_style = "github-dark"

# Without this, docutils reads a single-backtick span as a title reference and
# renders it in italics. Every page uses single backticks for identifiers and
# flags, so the default should be what the pages already mean.
default_role = "code"


todo_include_todos = True

current_file_dir = os.path.dirname(os.path.realpath(__file__))
CLAD_ROOT = current_file_dir + "/../../.."

with open(CLAD_ROOT + "/VERSION", "r") as f:
    version = f.read().strip()

# The sidebar header shows project and release together, so a stale release
# reads as the documentation's own version.
release = version

# Each diagram as tall as it needs to be; the default boxes every one of them
# into the same 500px. The width is settled in _static/clad.css, which also says
# why. A diagram wider than about 740px is scaled down to fit the text column,
# so keep them under that.
mermaid_height = "auto"
# Mermaid names each diagram after Date.now(), and every rule it emits is scoped
# to that name. Two diagrams rendered in the same millisecond collide, and the
# second one comes out unstyled and collapsed. Number them instead. The labels
# are set a little larger than mermaid's default, which is smaller than the body
# text beside it.
mermaid_init_config = {
    "startOnLoad": False,
    "deterministicIds": True,
    "themeVariables": {"fontSize": "18px"},
}

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
# Add latex physics package
mathjax3_config = {
    "loader": {"load": ["[tex]/physics"]},
    "tex": {"packages": {"[+]": ["physics"]}},
}
if os.environ.get("CLAD_BUILD_INTERNAL_DOCS"):
    html_extra_path = [CLAD_ROOT + "/build/docs/"]

    import shutil
    import subprocess

    CMAKE_CONFIGURE_COMMAND = (
        "mkdir {0}/build; cd {0}/build; cmake ../ "
        "-DClang_DIR=/usr/lib/llvm-18 -DLLVM_DIR="
        "/usr/lib/llvm-18 -DCLAD_ENABLE_DOXYGEN=ON "
        "-DCLAD_INCLUDE_DOCS=ON"
    ).format(CLAD_ROOT)
    # check_call, not call: these used to be able to fail and still leave a
    # green build, published with no internal documentation in it.
    subprocess.check_call(CMAKE_CONFIGURE_COMMAND, shell=True)

    INTERNAL_DOCS_DIR = "{0}/build/docs/internalDocs".format(CLAD_ROOT)
    RUN_DOXYGEN_COMMAND = "(cat doxygen.cfg; echo 'OUTPUT_DIRECTORY = .') | doxygen -"
    print(RUN_DOXYGEN_COMMAND)
    subprocess.check_call(RUN_DOXYGEN_COMMAND, shell=True, cwd=INTERNAL_DOCS_DIR)

    # html_extra_path publishes everything under build/docs, so without this the
    # files cmake and doxygen were driven by are served beside the documentation
    # they produced.
    for leftover in ("CMakeFiles", "Makefile", "cmake_install.cmake",
                     "doxygen.cfg"):
        path = os.path.join(INTERNAL_DOCS_DIR, leftover)
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
