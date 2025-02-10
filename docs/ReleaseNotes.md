Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.9. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.9?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-9 to clang-19


Forward Mode & Reverse Mode
---------------------------
* Improved differentiation support for complex mathematical expressions.
* Enhanced function overload handling in both forward and reverse modes.
* Enable differentiation when functions are defined in transparent contexts.


Forward Mode
------------
* Optimized forward mode differentiation for performance improvements.
* Added support for additional intrinsic functions.


Reverse Mode
------------
* Enhanced handling of control flow structures in reverse mode.
* Improved memory efficiency during differentiation.
* Support member calls with xvalue bases in the reverse mode.
* Initialize object adjoints using a copy of the original when copy constructor
  is available. Eg:
  ```
  std::vector<...> v{x, y, z};
  std::vector<...> _d_v{v}; // The length of the vector is preserved
  clad::zero_init(_d_v); // All elements are initialized to 0
  ```
  The new clad::zero_init function relies on the iterators to initialize the
  elements.


CUDA
----
* Introduced experimental support for differentiating CUDA kernels.
* Optimized CUDA differentiation to reduce compilation overhead.


Misc
----
* General improvements in documentation and code clarity.
* Various performance optimizations across Clad.
* Improved `DiffRequest` and `DynamicGraph` printing.


Fixed Bugs
----------

[767](https://github.com/vgvassilev/clad/issues/767)
[917](https://github.com/vgvassilev/clad/issues/917)
[1082](https://github.com/vgvassilev/clad/issues/1082)
[1112](https://github.com/vgvassilev/clad/issues/1112)
[1215](https://github.com/vgvassilev/clad/issues/1215)
[1216](https://github.com/vgvassilev/clad/issues/1216)

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

Vassil Vassilev (36)
petro.zarytskyi (12)
mcbarton (2)
PetroZarytskyi (2)
dependabot[bot] (1)
ToshitJain (1)
Rohith Suresh (1)
Abdelrhman Elrawy (1)
