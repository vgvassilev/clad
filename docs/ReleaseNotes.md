Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.6. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.6?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------
* Clad now works with clang-5.0, clang-6.0, clang-7.0, clang-8.0 and clang-9.0

Forward Mode & Reverse Mode
---------------------------
* Implement hessian matrices via the `clad::hessian` interface.

Reverse Mode
------------
* Reduce the quadratic cloning complexity to linear.
* Support variable reassignments pontentially depending on control flow.
* Support operators `+=`, `-=`, `*=`, `/=`, `,`, `++`, `--`.
* Allow assignments to array subscripts.
* Support nested assignments in expressions `a = b * ((c ? d : e) = f = g);`
* Enable differentiation of for-loops


Fixed Bugs
----------

[Issue 138](https://github.com/vgvassilev/clad/issues/138)

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

* Alexander Penev (19)
* Vassil Vassilev (15)
* Aleksandr Efremov (11)
* Shakhov Roman (2)
* Marco Foco (2)
* Jack Qiu (1)
