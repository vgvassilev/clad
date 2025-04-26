Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.10. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.10?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-10 to clang-20


Forward Mode & Reverse Mode
---------------------------
* Improved diagnostics for unsupported features​.
* Add custom derivatives for standard math functions.


Forward Mode
------------
* Add support of variadic functions.


Reverse Mode
------------

* Refined function analysis to mark const methods as non-differentiable​.
* Make to-be-recorded (TBR) analysis default.
* Automatically generate some constructor pullbacks.
* Implemented "store per change" for method bases to reduce tape usage.
* Enabled differentiation of global variables.
* Enabled differentiation for static member functions.
* Improved handling of object variables, differentiation more consistent across types​
* Improved support for default arguments in reverse mode​.
* Added primitive support for `std::unique_ptr<T>​`.
* Refactored handling of custom STL pullbacks​.
* Reduce tape usage for objects.


CUDA
----

* Added `CUDA_HOST_DEVICE` attributes to `zero_impl` and `zero_init​`.
* Created `ParmVarDecl` for local kernel variables in device pullbacks​.
* Reworked indexing of pullbacks to better handle CUDA differentiation​


Misc
----

* Support for LLVM 20, dropped LLVM 9.
* Updated GoogleTest to latest version​.
* General CI improvements: Ubuntu 22/24 support, dropped older configs​.
* Improved varied analysis.


Fixed Bugs
----------

[685](https://github.com/vgvassilev/clad/issues/685)
[760](https://github.com/vgvassilev/clad/issues/760)
[772](https://github.com/vgvassilev/clad/issues/772)
[800](https://github.com/vgvassilev/clad/issues/800)
[871](https://github.com/vgvassilev/clad/issues/871)
[879](https://github.com/vgvassilev/clad/issues/879)
[1009](https://github.com/vgvassilev/clad/issues/1009)
[1044](https://github.com/vgvassilev/clad/issues/1044)
[1095](https://github.com/vgvassilev/clad/issues/1095)
[1125](https://github.com/vgvassilev/clad/issues/1125)
[1262](https://github.com/vgvassilev/clad/issues/1262)
[1269](https://github.com/vgvassilev/clad/issues/1269)
[1270](https://github.com/vgvassilev/clad/issues/1270)
[1276](https://github.com/vgvassilev/clad/issues/1276)
[1294](https://github.com/vgvassilev/clad/issues/1294)
[1295](https://github.com/vgvassilev/clad/issues/1295)
[1302](https://github.com/vgvassilev/clad/issues/1302)
[1304](https://github.com/vgvassilev/clad/issues/1304)
[1353](https://github.com/vgvassilev/clad/issues/1353)

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

petro.zarytskyi (44)
Vassil Vassilev (21)
Max Andriychuk (11)
Christina Koutsou (7)
petro (1)
Shubhanjan-GGOD (1)
Rohan-T144 (1)
Petro Mozil (1)
Parth Arora (1)
Jayant (1)
Greg Hazel (1)
Errant (1)
Abdelrhman Elrawy (1)
