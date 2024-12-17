Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.8. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.8?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad works with clang-8 to clang-18.


Forward Mode & Reverse Mode
---------------------------
* Support `std::array`, `std::vector`.

Forward Mode
------------
* Support `std::tie`, `std::atan2` and `std::acos`.

Reverse Mode
------------
* Support `std::initializer_list` and `sizeof`.
* Support pointer-valued functions.
* Support range-based for loops.

CUDA
----
* Add support of CUDA device pullbacks.
* Enable computation of CUDA global kernels derivative.


Misc
----
* Enable immediate evaluation mode (`consteval` and `constexpr`) with a new
  clad mode `clad::immediate_mode`.
* Make `clad::CladFunction` and `clad::array_ref` constexpr.
* Support operators defined outside of classes.
* Add Varied analysis to the reverse mode.
* Add support for `Kokkos::View`, `Kokkos::deep_copy`, Kokkos::resize and
  `parallel_for` in reverse mode.
* Add support for `Kokkos::parallel_for`, `Kokkos::fence`, `Kokkos::View` and
  `Kokkos::parallel_reduce` in forward mode.


Fixed Bugs
----------

[472](https://github.com/vgvassilev/clad/issues/472)
[480](https://github.com/vgvassilev/clad/issues/480)
[527](https://github.com/vgvassilev/clad/issues/527)
[682](https://github.com/vgvassilev/clad/issues/682)
[684](https://github.com/vgvassilev/clad/issues/684)
[830](https://github.com/vgvassilev/clad/issues/830)
[855](https://github.com/vgvassilev/clad/issues/855)
[1000](https://github.com/vgvassilev/clad/issues/1000)
[1019](https://github.com/vgvassilev/clad/issues/1019)
[1033](https://github.com/vgvassilev/clad/issues/1033)
[1049](https://github.com/vgvassilev/clad/issues/1049)
[1057](https://github.com/vgvassilev/clad/issues/1057)
[1070](https://github.com/vgvassilev/clad/issues/1070)
[1071](https://github.com/vgvassilev/clad/issues/1071)
[1081](https://github.com/vgvassilev/clad/issues/1081)
[1087](https://github.com/vgvassilev/clad/issues/1087)
[1151](https://github.com/vgvassilev/clad/issues/1151)
[1162](https://github.com/vgvassilev/clad/issues/1162)

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

petro.zarytskyi (30)
Vassil Vassilev (22)
Atell Krasnopolski (22)
Christina Koutsou (17)
Mihail Mihov (12)
ovdiiuv (7)
kchristin (6)
Vipul Cariappa (3)
Alexander Penev (3)
mcbarton (1)
infinite-void-16 (1)
fsfod (1)
Vaibhav Thakkar (1)
Max Andriychuk (1)
Infinite Void (1)
Austeja (1)
