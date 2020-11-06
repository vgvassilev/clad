---
title: "Open Projects"
layout: textlay
excerpt: "Projects"
sitemap: false
permalink: /open_projects
---

<nav>
  <h4>Table of Contents</h4>
  * this unordered seed list will be replaced by toc as unordered list
  {:toc}
</nav>


# Add numerical differentiation support in clad

## Description

In mathematics and computer algebra, automatic differentiation (AD) is a set of
techniques to numerically evaluate the derivative of a function specified by a
computer program. Automatic differentiation is an alternative technique to
Symbolic differentiation and Numerical differentiation (the method of finite
differences). Clad is based on Clang which will provide the necessary facilities
for code transformation. The AD library is able to differentiate non-trivial
functions, to find a partial derivative for trivial cases and has good unit test
coverage.

Currently, clad cannot differentiate declared-but-not-defined functions.
In that case it issues an error. Instead, clad should fall back to its future
numerical differentiation facilities.

## Task ideas and expected results

Implement numerical differentiation support in clad. It should be available
through a dedicated interface (for example `clad::num_differentiate`).
The new functionality should be connected to the forward mode automatic
differentiation. If time permits, a prototype of configurable error estimation
for the numerical differentiation should be implemented. The student should be
prepared to write a progress report and present the results.

# Infrastructure

## Improve Cling's packaging system cpt

Cling has a flexible tool which can build and package binaries. It is
implemented in python.

Currently it has an issue deb package creation. The tool calls shell commands
such as  `mv` and  `wget` which should be replaced with their proper python
versions.


## Cling GitHub PR Code Coverage

See [this example](https://github.com/vgvassilev/clad/blob/master/.travis.yml#L824).

## Automatically upload nightlies to a special release tag

See [this example](https://github.com/vgvassilev/cling/blob/master/.travis.yml#L192-L214).
