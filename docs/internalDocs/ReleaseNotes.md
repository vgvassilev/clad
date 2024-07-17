Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.6. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.6?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-8 to clang-18


Forward Mode & Reverse Mode
---------------------------
* Add support for simple lambda functions


Forward Mode
------------
* Support condition declarations and assignments
* Enable logical operators in for loops
* Support top level custom derivatives in vector mode
* Add support for assignments in while-loops
* Support for initializer_list
* Support for const_cast
* Add support for `std::string` variables


Reverse Mode
------------
* Improve consistency in the tape usage
* Support pointer reference parameters
* Remove redundant goto/label statements from the generated code

Misc
----
* Improved CMake infrastructure via `AddClad.cmake`
* Remove unnecessary clad::array_ref usages
* Add support for computing only the diagonal hessian entries
* Basic support for custom constructor pushforward functions


Fixed Bugs
----------

[273](https://github.com/vgvassilev/clad/issues/273)
[352](https://github.com/vgvassilev/clad/issues/352)
[509](https://github.com/vgvassilev/clad/issues/509)
[789](https://github.com/vgvassilev/clad/issues/789)
[874](https://github.com/vgvassilev/clad/issues/874)
[899](https://github.com/vgvassilev/clad/issues/899)
[908](https://github.com/vgvassilev/clad/issues/908)
[911](https://github.com/vgvassilev/clad/issues/911)
[913](https://github.com/vgvassilev/clad/issues/913)
[922](https://github.com/vgvassilev/clad/issues/922)
[927](https://github.com/vgvassilev/clad/issues/927)
[951](https://github.com/vgvassilev/clad/issues/951)
[965](https://github.com/vgvassilev/clad/issues/965)
[972](https://github.com/vgvassilev/clad/issues/972)
[974](https://github.com/vgvassilev/clad/issues/974)
[978](https://github.com/vgvassilev/clad/issues/978)

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

Vaibhav Thakkar (16)
petro.zarytskyi (13)
Vassil Vassilev (13)
Atell Krasnopolski (11)
parth-07 (3)
dependabot[bot] (3)
Rohan Julka (2)
Jonas Rembser (2)
PetroZarytskyi (1)
Maxxxx (1)
Max Andriychuk (1)
Alexander Penev (1)
