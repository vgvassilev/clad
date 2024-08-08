Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.7. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.7?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-8 to clang-18


Forward Mode & Reverse Mode
---------------------------
* Add propagators for `__builtin_pow` and `__builtin_log`
* Support range-based for loops
* Improve diagnostics clarity

Forward Mode
------------
* Advance support of frameworks such as Kokkos
* Support `std::array`

Reverse Mode
------------
* Support non_differentiable attribute


Fixed Bugs
----------

[46](https://github.com/vgvassilev/clad/issues/46)
[381](https://github.com/vgvassilev/clad/issues/381)
[479](https://github.com/vgvassilev/clad/issues/479)
[525](https://github.com/vgvassilev/clad/issues/525)
[717](https://github.com/vgvassilev/clad/issues/717)
[723](https://github.com/vgvassilev/clad/issues/723)
[829](https://github.com/vgvassilev/clad/issues/829)
[979](https://github.com/vgvassilev/clad/issues/979)
[983](https://github.com/vgvassilev/clad/issues/983)
[986](https://github.com/vgvassilev/clad/issues/986)
[988](https://github.com/vgvassilev/clad/issues/988)
[1005](https://github.com/vgvassilev/clad/issues/1005)


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

petro.zarytskyi (11)
Vassil Vassilev (11)
Atell Krasnopolski (5)
Vaibhav Thakkar (2)
Mihail Mihov (2)
ovdiiuv (1)
Rohan Julka (1)
Max Andriychuk (1)
