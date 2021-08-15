Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.9. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.9?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-5.0 to clang-13-rc1


Forward Mode & Reverse Mode
---------------------------

* Add support for differentiating functor-like objects.
* Preserve the type qualifiers in the derived function.
* Develop initial support for differentiation of CUDA code.
* Improve the doxygen-style documentation.


Forward Mode
------------

* Add support for differentiating while and do-while statements
* Add switch statement differentiation support.
* Add array differentiation support.
* Allow the user to specify an array index as a independent variable. For
  instance, `clad::differentiate(f, "p[1]");`.


Reverse Mode
------------

* Extend the array differentiation support. See more in the
  [demo]https://github.com/vgvassilev/clad/blob/v0.9/demos/Arrays.cpp).


Build System
------------

* Add cmake variables to control the locations where find_package discovers
  LLVM and Clang: `LLVM_CONFIG_EXTRA_PATH_HINTS` and
  `Clang_CONFIG_EXTRA_PATH_HINTS` respectively.

Fixed Bugs
----------

* Fix memory leaks in `clad::Tape`.
* Fix bug in the `clad::Tape::size()`.
* Fix codegen Error for class function differentiation
  ([139](https://github.com/vgvassilev/clad/issues/139))


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

* Baidyanath Kundu (21)
* Parth Arora (21)
* Vassil Vassilev (3)
* Garima Singh (3)
* axmat (1)
* Ioana Ifrim (1)
* Alexander Penev (1)
