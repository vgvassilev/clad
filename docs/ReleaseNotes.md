Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.8. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.8?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------
* Clad now works with clang-5.0 to clang-12

Forward Mode & Reverse Mode
---------------------------
* Implement #pragma clad ON/OFF/DEFAULT to control regions where clad is active
* Support member functions with qualifiers in differentiation calls
* Add getCode() interface for interactive use
* Add support for using casts, `*` and `&` operators. For example:
  `clad::differentiate(*&ptr_to_ptr, "...");`

Misc
----
* Add support for clang-11.1.0
* Add support for clang-12


Fixed Bugs
----------

* Fixed several crashes in reverse mode.


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

* Vassil Vassilev (14)
* Baidyanath Kundu (10)
* Garima Singh (7)
* Alexander Penev (5)
* Pratyush Das (3)
* Parth Arora (2)
* Ioana Ifrim (2)
* Oksana Shadura (1)
* Alex Efremov (1)
