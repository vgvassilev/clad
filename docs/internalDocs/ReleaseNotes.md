Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.0. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.0?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-5.0 to clang-14


Forward Mode & Reverse Mode
---------------------------
* Add support for pushforward- and pullback-style functions  which allow to
  accumulate the results from the AD when required to correctly compute
  derivatives when arguments are passed by reference or pointers


Forward Mode
------------
* Add support for member variables in functors
* Add basic support for virtual functions
* Add support for reference arguments
* Add basic support for AD of class types wrt scalars
* Add support for member functions, pointers, overloaded operators, pointer
  arithmetic, nullptr, sizeof and pseudo objects,



Reverse Mode
------------
* Add support for `while` and `do-while` statements
* Add initial support for AD of user-defined types allowing to
  differentiate scalar types wrt user-defined types


CUDA
----
* Add forward mode support for basic CUDA programs. More can be seen
  [here](https://github.com/vgvassilev/clad/tree/v1.0/test/CUDA)


Error Estimation
----------------
* Developed an error estimation framework to perform AD-based error estimation.
  The new facility is available via the `clad::estimate_error` interface.
  See more in this [demo](https://github.com/vgvassilev/clad/tree/v1.0/demos/ErrorEstimation)

Misc
----
* Developed user documentation available at [clad.readthedocs.io](https://clad.readthedocs.io/en/latest/)
* Developed developers documentation available at [doxygen](https://clad.readthedocs.io/en/latest/internalDocs/html/index.html)
* Implement a fallback to numerical differentiation if Clad cannot differentiate
  a given function. To disable this behavior, please compile your programs with
  the `-DCLAD_NO_NUM_DIFF`.
* Add benchmarking infrastructure based on google benchmark
* Add integration with Enzyme via `clad::gradient<clad::opts::use_enzyme>(...)`

Fixed Bugs
----------

[28](https://github.com/vgvassilev/clad/issues/28)
[281](https://github.com/vgvassilev/clad/issues/281)
[353](https://github.com/vgvassilev/clad/issues/353)
[368](https://github.com/vgvassilev/clad/issues/368)
[386](https://github.com/vgvassilev/clad/issues/386)
[387](https://github.com/vgvassilev/clad/issues/387)
[393](https://github.com/vgvassilev/clad/issues/393)
[440](https://github.com/vgvassilev/clad/issues/440)


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

Parth Arora (65)
Vassil Vassilev (49)
Garima Singh (17)
Baidyanath Kundu (13)
Nirhar (12)
Ioana Ifrim (9)
Alexander Penev (4)
RohitRathore1 (1)
David (1)

