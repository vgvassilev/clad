Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 2.6. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 2.6?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-14 to clang-23


Forward Mode & Reverse Mode
---------------------------
*

Forward Mode
------------
*

Reverse Mode
------------
*

CUDA
----
*

Error Estimation
----------------
*

Misc
----
* `clad::immediate_mode` is gone. Clad decides for itself whether a derivative
  is needed while the program compiles, from where the call to
  `clad::differentiate` sits, so code passing the option should simply drop it.
  Every mode now gets it, which makes `clad::gradient` usable in an immediate
  context when compiled as C++26.

Fixed Bugs
----------

[XXX](https://github.com/vgvassilev/clad/issues/XXX)

 <!---Get release bugs. Check for close, fix, resolve
 git log v2.5..master | grep -i "close" | grep '#' | sed -E 's,.*\#([0-9]*).*,\[\1\]\(https://github.com/vgvassilev/clad/issues/\1\),g' | sort -t'[' -k2,2n
 --->

<!--- https://github.com/vgvassilev/clad/issues?q=is%3Aissue%20state%3Aclosed%20closed%3A%3E2025-10-01 --->

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

<!---Find contributor list for this release
 git log --pretty=format:"%an"  v2.5...master | sort | uniq -c | sort -rn | sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
