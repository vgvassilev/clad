Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.5. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.5?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-8 to clang-18


Forward Mode & Reverse Mode
---------------------------
* Add support for C-style memory alloc and free


Reverse Mode
------------
* Replace array_ref with pointers in gradient signature
* Initial support for new and delete operations in reverse mode


Error Estimation
----------------
* Only track sizes of independent arrays
* Remove .size() from error estimation
* Simplify error estimation by removing `_EERepl_` and `_delta_`

Misc
----
* Teach binder to use the newest available version of clad
* Simplify pullback calls by replacing `_grad`/`_r` pairs with single `_r`
  variables
* Delay the differentiation process until the end of TU -- Clad now can operate
  just like clang and visit the entire translation unit to construct a precise
  differentiation plan
* Remove extra lines generated when using clad::array or array_ref
* Added timings report if `-ftime-report` flag is enabled

Fixed Bugs
----------

[248](https://github.com/vgvassilev/clad/issues/248)
[350](https://github.com/vgvassilev/clad/issues/350)
[704](https://github.com/vgvassilev/clad/issues/704)
[715](https://github.com/vgvassilev/clad/issues/715)
[765](https://github.com/vgvassilev/clad/issues/765)
[769](https://github.com/vgvassilev/clad/issues/769)
[790](https://github.com/vgvassilev/clad/issues/790)
[792](https://github.com/vgvassilev/clad/issues/792)
[798](https://github.com/vgvassilev/clad/issues/798)
[805](https://github.com/vgvassilev/clad/issues/805)
[854](https://github.com/vgvassilev/clad/issues/854)
[865](https://github.com/vgvassilev/clad/issues/865)
[867](https://github.com/vgvassilev/clad/issues/867)
[886](https://github.com/vgvassilev/clad/issues/886)
[887](https://github.com/vgvassilev/clad/issues/887)
[890](https://github.com/vgvassilev/clad/issues/890)
[897](https://github.com/vgvassilev/clad/issues/897)

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

petro.zarytskyi (30)
Vaibhav Thakkar (24)
Vassil Vassilev (21)
mcbarton (6)
Mihail Mihov (4)
dependabot[bot] (2)
Atell Krasnopolski (2)
Alexander Penev (2)
sauravUppoor (1)
kchristin22 (1)
Warren Jacinto (1)
Jonas Hahnfeld (1)
Deeptendu Santra (1)
Christina Koutsou (1)
