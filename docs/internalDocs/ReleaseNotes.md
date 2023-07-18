Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.2. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.2?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-5.0 to clang-16


Forward Mode
------------
* Add experimental support for forward vector mode.
* Improve support of comma expressions.

Reverse Mode
------------
*  Add pushforwards for `std::floor` and `std::ceil`.

Misc
----
* Fill `clad::array` with 0s when assigned an empty brace-init list.
* Improve documentation
* Improve AD function interfaces with bitmasked options. For example:
  `clad::differentiate<<clad::order::first, clad::opts::vector_mode>(f)` will
  be equivalent to `clad::differentiate<<1, clad::opts::vector_mode>(f)` and
  will request the first order derivative of `f` in forward vector mode.

Fixed Bugs
----------

[218](https://github.com/vgvassilev/clad/issues/218)
[395](https://github.com/vgvassilev/clad/issues/395)
[521](https://github.com/vgvassilev/clad/issues/521)
[523](https://github.com/vgvassilev/clad/issues/523)
[541](https://github.com/vgvassilev/clad/issues/541)
[566](https://github.com/vgvassilev/clad/issues/566)
[573](https://github.com/vgvassilev/clad/issues/573)
[582](https://github.com/vgvassilev/clad/issues/582)

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

Jonas Hahnfeld (27)
Vaibhav Thakkar (18)
Ris-Bali (5)
Garima Singh (3)
Vassil Vassilev (2)
Rishabh Bali (2)
vidushi (1)
petro.zarytskyi (1)
daemondzh (1)
ShounakDas101 (1)
Prajwal S N (1)
PetroZarytskyi (1)
Daemond (1)

