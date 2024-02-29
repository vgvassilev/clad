Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.4. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.4?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-7 to clang-17


Forward Mode & Reverse Mode
---------------------------
* Improve handling of char and string literals


Reverse Mode
------------
* Add support for differentiating switch statements
* Supportpassing pointers as call arguments
* Fix pointer arithmetic for array types


Misc
----
* Support BUILD_SHARED_LIBS=On

Fixed Bugs
----------

[300](https://github.com/vgvassilev/clad/issues/300)
[313](https://github.com/vgvassilev/clad/issues/313)
[636](https://github.com/vgvassilev/clad/issues/636)
[735](https://github.com/vgvassilev/clad/issues/735)
[748](https://github.com/vgvassilev/clad/issues/748)
[753](https://github.com/vgvassilev/clad/issues/753)
[774](https://github.com/vgvassilev/clad/issues/774)


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

Vassil Vassilev (9)
Vaibhav Thakkar (6)
maximusron (1)
bedupako12mas (1)
Parth (1)
Krishna Narayanan (1)
Aaron  Jomy (1)
