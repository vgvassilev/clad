Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.7. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.7?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------
* Clad now works with clang-5.0 to clang-12

Forward Mode & Reverse Mode
---------------------------
* Implement hessian matrices via the `clad::jacobian` interface.


Fixed Bugs
----------

* Fixed the discovery of llvm in special builds with clang and libcxx.


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

* Roman Shakhov (3)
* Philippe Canal (2)
* Alexander Penev (2)
* Vassil Vassilev (1)
