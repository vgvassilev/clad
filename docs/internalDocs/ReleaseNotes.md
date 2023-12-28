Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.3. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.3?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-5.0 to clang-17


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
*

Fixed Bugs
----------

[XXX](https://github.com/vgvassilev/clad/issues/XXX)

 <!---Get release bugs
 git log v1.2..master | grep 'Fixes|Closes'
 --->

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

<!---Find contributor list for this release
 git log --pretty=format:"%an"  v1.2...master | sort | uniq -c | sort -rn |\
   sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
