Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 2.0. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 2.0?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-10 to clang-20


Forward Mode & Reverse Mode
---------------------------

* Improved handling of non-differentiable annotated types
* Support `static_assert`.
* Improved to-be-recorded (TBR) analysis.


Forward Mode
------------

* Added support for `Useful Analysis` for more efficient derivative computation.
* Custom derivative resolution now leverages Clang's `CallExpr` infrastructure
  for correctness and compatibility.
* Improved diagnostics for mismatched custom derivative signatures.
* Enhanced handling of real-valued parameters only.


Reverse Mode
------------

* Enabled static scheduling for constructor pullbacks.
* Reworked `reverse_forw` naming and logic consistency.
* Improved support for `std::shared_ptr`, `std::weak_ptr`, and complex
  value-type parameters.
* Support added for custom derivatives for STL iterators.
* Static-only scheduling for pullbacks is now the default (excluding Hessians
  and special cases like error estimation).
* Improved support for rvalue references.


CUDA
----

* Tape now supports CUDA builds and works with nvcc.
* Recalculation support for CUDA built-in index functions (threadIdx, etc.)


Error Estimation
----------------

* The error estimation framework has been updated to maintain singleton objects
  per request, ensuring more consistent handling.


Misc
----

* More descriptive stack traces for diagnostics.
* Improved support for polymorphism and virtual functions in user demos.
* A benchmark comparison script has been added.
* The SmallPT ray tracer demo has been revised and fixed.
* Added -version and improved --help usability.
* Godbolt link included in documentation.
* Reworked time reporting -- detailed performance information is shown when the
  env variable `CLAD_ENABLE_TIMING` is set or passed `-ftime-report`.
* Handle custom derivatives ahead of the differentiation process improving the
  overall system robustness.


Fixed Bugs
----------

[1454](https://github.com/vgvassilev/clad/issues/1454)
[1452](https://github.com/vgvassilev/clad/issues/1452)
[1451](https://github.com/vgvassilev/clad/issues/1451)
[1437](https://github.com/vgvassilev/clad/issues/1437)
[1429](https://github.com/vgvassilev/clad/issues/1429)
[1423](https://github.com/vgvassilev/clad/issues/1423)
[1392](https://github.com/vgvassilev/clad/issues/1392)
[1383](https://github.com/vgvassilev/clad/issues/1383)
[1371](https://github.com/vgvassilev/clad/issues/1371)
[1369](https://github.com/vgvassilev/clad/issues/1369)
[1366](https://github.com/vgvassilev/clad/issues/1366)
[1123](https://github.com/vgvassilev/clad/issues/1123)
[985](https://github.com/vgvassilev/clad/issues/985)
[793](https://github.com/vgvassilev/clad/issues/793)
[751](https://github.com/vgvassilev/clad/issues/751)
[665](https://github.com/vgvassilev/clad/issues/665)


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

petro.zarytskyi (57)
Vassil Vassilev (20)
ovdiiuv (4)
Vipul Cariappa (2)
dependabot[bot] (2)
aditimjoshi (2)
Rohan-T144 (1)
Rohan Timmaraju (1)
mcbarton (1)
Maki Arima (1)
Christina Koutsou (1)
Abhinav Kumar (1)
Abdelrhman Elrawy (1)
