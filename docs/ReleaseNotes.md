Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.5. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.5?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------
* clang-5.0

Forward & Reverse Mode
------------
* Extend the way to specify a dependent variables. Consider function,
  `double f(double x, double y, double z) {...}`, `clad::differentiate(f, "z")`
  is equivalent to `clad::differentiate(f, 2)`. `clad::gradient(f, "x, y")`
  differentiates with respect to `x` and `y` but not `z`. The gradient results
  are stored in a `_result` parameter in the same order as `x` and `y` were
  specified. Namely, the result of `x` is stored in `_result[0]` and the result
  of `y` in `_result[1]`. If we invert the arguments specified in the string to
  `clad::gradient(f, "y, x")` the results will be stored inversely.
* Enable recursive differentiation.
* Support single- and multi-dimensional arrays -- works for arrays with constant
  size like `double A[] = {1, 2, 3};`, `double A[3];` or `double A[1][2][3][4];`

Reverse Mode
------------
* Support variable reassignments. For example,
```cpp
double f(double x, double y) {
  double a = x * x;
  double b = y * y;
  double c = a + b;
  return c;
}
```

Misc
----
* Add coverity static analyzer to the pull request builds.
* Fix found by coverity issues.
* Improved README.

Fixed Bugs
----------

[Issue 77](https://github.com/vgvassilev/clad/issues/77)
[Issue 105](https://github.com/vgvassilev/clad/issues/105)

<!---Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' --->
<!---Get release bugs
git log v0.4..master | grep 'Fixes' | \
  s,^.*([0-9]+).*$,[\1]\(https://github.com/vgvassilev/clad/issues/\1\),' | uniq
--->
<!---Standard MarkDown doesn't support neither variables nor <base>
[Issue XXX](https://github.com/vgvassilev/clad/issues/XXX)
--->


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits):
* Aleksandr Efremov(7)
* Vassil Vassilev (6)
* Oksana Shadura (2)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.4...master | sort | uniq -c | sort -rn
--->
