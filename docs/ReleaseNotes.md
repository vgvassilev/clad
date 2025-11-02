Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 2.2. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 2.2?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-11 to clang-21
* Removed unused coverage libraries for faster CI builds.
* Fixed Python 3.12 linking issue in LLVM 16 installs on macOS.
* macOS ARM CI updated to macOS 26.


Forward Mode & Reverse Mode
---------------------------

* Major internal cleanup and generalization of differentiation pipelines.
* Unified initialization logic for adjoints and original variables.
* Improved compatibility across pointer, tensor, and reference types.
* Added support for conversion operators and `std::reference_wrapper`.
* Enhanced handling of initialization lists, pseudo-destructors, and complex
  expressions.


Forward Mode
------------

* OpenMP Support (Experimental): Introduced basic OpenMP support for forward
  mode differentiation.
* Simplified adjoint and initializer handling for forward-pass variables.

Reverse Mode
------------

* Added reverse mode checkpointing for loops, improving memory efficiency in
  long iterations.
* Elidable Reverse Passes: Introduced `elidable_reverse_forw` attribute to skip
  redundant reverse passes for trivially invertible functions.
* `constructor_reverse_forw` now supports static scheduling and
  `elidable_reverse_forw`.
* Added support for `CompoundLiteralExpr` and improved differentiation through
  compound expressions.
* Simplified handling of deallocations and memory operations via attributes.
* Improved function differentiation sequence by resolving pullbacks before
  argument differentiation.
* Unified differentiation order for pointer and reference types.
* Optimized unary operator simplification (removed redundant &*_d_x patterns).


CUDA
----

* Extended Thrust differentiation support:
  - `thrust::reduce_by_key`
  - `thrust::sort_by_key`
  - `thrust::adjacent_difference`
  - segmented scans and prefix-sum operations
* Added thrust::device_vector support.
* Introduced BoW (Bag-of-Words) logistic regression demo using Thrust.
* Replaced iterator-based std::move with CUDA-safe clad::move.
* Added generic functor support for Thrust transform.


Misc
----

* Added thread-safe tape access functions:
  - Zero overhead in single-threaded mode.
  - Controlled locking for multithreaded differentiation.
* Improved handling of ill-formed code by skipping Clad runs when Clang
  compilation fails.
* Refined diagnostic messages and simplified deallocation functions through
  attribute-based design.
* Updated testing and build infrastructure:
  - Added STL test coverage (starting with <cmath>).
  - Cleaned up coverage configuration.
  - Enhanced CI stability and performance.

Fixed Bugs
----------

[679](https://github.com/vgvassilev/clad/issues/679)
[1413](https://github.com/vgvassilev/clad/issues/1413)
[1482](https://github.com/vgvassilev/clad/issues/1482)
[1496](https://github.com/vgvassilev/clad/issues/1496)
[1521](https://github.com/vgvassilev/clad/issues/1521)
[1522](https://github.com/vgvassilev/clad/issues/1522)

 <!---Get release bugs. Check for close, fix, resolve
 git log v2.1..master | grep -i "close" | grep '#' | sed -E 's,.*\#([0-9]*).*,\[\1\]\(https://github.com/vgvassilev/clad/issues/\1\),g' | sort -t'[' -k2,2n
 --->

<!--- https://github.com/vgvassilev/clad/issues?q=is%3Aissue%20state%3Aclosed%20closed%3A%3E2025-10-01 --->

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

Petro Zarytskyi (33; reverse mode, loop checkpointing, elidable reverse passes)
Abdelrhman Elrawy (9; CUDA/Thrust derivatives, demos, logistic regression)
Vassil Vassilev (9; compiler integration, CI, and build infrastructure)
Matthew Barton (2; macOS and Python 3.12 CI fixes)
Aditi Joshi (1; thread-safe tape access implementation)
Max Andriychuk (1; analyses)
Errant (1; OpenMP differentiation support)

<!---Find contributor list for this release
 git log --pretty=format:"%an"  v2.1...master | sort | uniq -c | sort -rn | sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
