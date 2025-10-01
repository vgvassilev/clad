Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 2.1. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 2.1?
========================

Clad 2.1 introduces major advancements in reverse mode differentiation, bringing
smarter handling of loops, assignments, and method calls, alongside the new
`clad::restore_tracker` for functions that modify their inputs. Forward mode
gains static scheduling for Hessians and higher-order derivatives, while CUDA
support expands with custom derivatives for key Thrust algorithms such as
`reduce`, `transform`, and `transform_reduce`, plus optimizations that reduce
unnecessary GPU atomics. The release also strengthens error estimation,
simplifies adjoint initialization, improves tape efficiency, and enhances
diagnostics. With a migration to C++17, support extended up to clang-21, and
numerous bug fixes, Clad 2.1 delivers faster, safer, and more reliable automatic
differentiation across CPU and GPU workflows.

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-11 to clang-21.
* Switch to C++17 standard.


Forward Mode & Reverse Mode
---------------------------

* Improve support for differentiation of function/method calls:
  - Move call/argument differentiation into unified helpers.
  - Improve handling of base initializers, delegating constructors, and method
    bases.
* Optimizations in Tape-Based Recording (TBR):
  - Enable TBR for pointer arithmetic, nested calls, local variables, loops,
    and constructors.
  - Improve analysis of pointers, nested derivatives, and division denominators.
  - Remove redundant or unsafe `getZeroInit` initializations in aggregates and
    init lists.
* Statically schedule higher-order derivatives for Hessians.


Forward Mode
------------
*

Reverse Mode
------------

* Better handling of pullbacks and reverse_forw:
  - Avoid generating empty or redundant pullbacks.
  - Improve type consistency in reverse_forw.
  - Introduce `clad::restore_tracker` to support functions modifying their
    arguments.
* Improve differentiation ordering (differentiate method bases before
  arguments).
* Simplify assignments when LHS and RHS are independent.
* Support for for-loops without increments.
* Add optimizations to avoid unnecessary storage of unused return values.
* Consider unions differentiable.
* Support for not differentiating w.r.t. const references.
* Numerous fixes in `reverse_forw` handling of types and aggregates.


CUDA
----
* Add custom derivatives for several Thrust algorithms:
  - `thrust::reduce`, `thrust::transform`, `thrust::transform_reduce`,
    `thrust::copy`, `thrust::inner_product`.
* Improve handling of CUDA atomics:
  - Avoid atomics for injective index computations.
  - Added liveness analysis for removing unnecessary atomics.


Error Estimation
----------------

* Reverse mode error estimation now uses pullbacks consistently.
* Fix `_final_error` value propagation .

Misc
----

* Improve diagnostics for propagator signature mismatches.
* Full qualification logic for generated code.
* Better handling of consteval functions, decl refs, and unused parameters.
* Remove unused STL derivatives and redundant infrastructure.
* CI improvements:
  - Added valgrind test runs.
  - Updated MacOS runners, godbolt build (clang 21), and codecov handling.

* Documentation: Update README to clarify plugin usage, C++14 requirement, and
  plugin args.

Fixed Bugs
----------

[428](https://github.com/vgvassilev/clad/issues/428)
[691](https://github.com/vgvassilev/clad/issues/691)
[752](https://github.com/vgvassilev/clad/issues/752)
[768](https://github.com/vgvassilev/clad/issues/768)
[1274](https://github.com/vgvassilev/clad/issues/1274)
[1301](https://github.com/vgvassilev/clad/issues/1301)
[1346](https://github.com/vgvassilev/clad/issues/1346)
[1349](https://github.com/vgvassilev/clad/issues/1349)
[1411](https://github.com/vgvassilev/clad/issues/1411)
[1419](https://github.com/vgvassilev/clad/issues/1419)
[1436](https://github.com/vgvassilev/clad/issues/1436)
[1445](https://github.com/vgvassilev/clad/issues/1445)
[1457](https://github.com/vgvassilev/clad/issues/1457)
[1466](https://github.com/vgvassilev/clad/issues/1466)
[1469](https://github.com/vgvassilev/clad/issues/1469)
[1472](https://github.com/vgvassilev/clad/issues/1472)
[1473](https://github.com/vgvassilev/clad/issues/1473)
[1490](https://github.com/vgvassilev/clad/issues/1490)
[1497](https://github.com/vgvassilev/clad/issues/1497)
[1498](https://github.com/vgvassilev/clad/issues/1498)
[1555](https://github.com/vgvassilev/clad/issues/1555)
[1557](https://github.com/vgvassilev/clad/issues/1557)
[1559](https://github.com/vgvassilev/clad/issues/1559)
[1560](https://github.com/vgvassilev/clad/issues/1560)
[1565](https://github.com/vgvassilev/clad/issues/1565)
[1573](https://github.com/vgvassilev/clad/issues/1573)
[1581](https://github.com/vgvassilev/clad/issues/1581)
[1582](https://github.com/vgvassilev/clad/issues/1582)


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

PetroZarytskyi (60; major ad/tbr work)
Vassil Vassilev (30; infrastructure, ci, compiler integration)
Abdelrhman Elrawy (7; cuda/thrust support)
Max Andriychuk (7; va/tbr infrastructure, cfg/loop analysis, cuda atomics)
aditimjoshi (4; tape improvements, benchmarks)
Vipul Cariappa (1; godbolt integration)
Timo Nicolai (1; docs, usage improvements)
Rohan Timmaraju (1; tensor-specific support)
mcbarton (1; ci, runners)
Jonas Rembser (1; build fixes)
Christina Koutsou (1; nondiff structs support)
