Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.3. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.3?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-7 to clang-17


Forward Mode
------------
* Make forward vector mode more robust:
  * Implement dedicated clad::matrix class
  * Add support for array arguments
  * Add support for call expressions
* Add support for the 'non_differentiable' attribute

Reverse Mode
------------
* Fix computation of higher order functions
* Introduce experimental To-Be-Recorded Analysis in Clad
* Improve storing of LHS/RHS in multiplication/division operators
* Add initial support for pointers
* Improve the overall performance by reducing the tape storage

Misc
----
* Add support for `std::min`, `std::max` and `std::clamp` functions
* Fix strong symbol definitions in Differentiator.h

Fixed Bugs
----------

[49](https://github.com/vgvassilev/clad/issues/49)
[86](https://github.com/vgvassilev/clad/issues/86)
[197](https://github.com/vgvassilev/clad/issues/197)
[275](https://github.com/vgvassilev/clad/issues/275)
[314](https://github.com/vgvassilev/clad/issues/314)
[429](https://github.com/vgvassilev/clad/issues/429)
[439](https://github.com/vgvassilev/clad/issues/439)
[441](https://github.com/vgvassilev/clad/issues/441)
[465](https://github.com/vgvassilev/clad/issues/465)
[606](https://github.com/vgvassilev/clad/issues/606)
[620](https://github.com/vgvassilev/clad/issues/620)
[650](https://github.com/vgvassilev/clad/issues/650)
[655](https://github.com/vgvassilev/clad/issues/655)
[660](https://github.com/vgvassilev/clad/issues/660)
[664](https://github.com/vgvassilev/clad/issues/664)
[667](https://github.com/vgvassilev/clad/issues/667)
[669](https://github.com/vgvassilev/clad/issues/669)
[672](https://github.com/vgvassilev/clad/issues/672)
[676](https://github.com/vgvassilev/clad/issues/676)
[681](https://github.com/vgvassilev/clad/issues/681)
[687](https://github.com/vgvassilev/clad/issues/687)
[689](https://github.com/vgvassilev/clad/issues/689)

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

Vaibhav Thakkar (43)
Vassil Vassilev (26)
Alexander Penev (8)
petro.zarytskyi (6)
dependabot[bot] (4)
Parth (2)
Rishabh Bali (1)
QuillPusher (1)
Krishna-13-cyber (1)
daemondzh (1)
Aaron Jomy (1)
