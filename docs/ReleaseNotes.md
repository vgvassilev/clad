Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.1. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Cla in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.1?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------
* Upgrade to clang 5.0.

Misc
----
* Implement forward automatic differentiation mode to compute derivatives of
arbitrary C/C++ functions.

Experimental Features
---------------------
* Implement reverse automatic differentiation mode to compute gradients of
arbitrary C/C++ functions.

Fixed Bugs
----------

[Issue 7](https://github.com/vgvassilev/clad/issues/7)
[Issue 8](https://github.com/vgvassilev/clad/issues/8)
[Issue 9](https://github.com/vgvassilev/clad/issues/9)
[Issue 13](https://github.com/vgvassilev/clad/issues/13)
[Issue 14](https://github.com/vgvassilev/clad/issues/14)
[Issue 16](https://github.com/vgvassilev/clad/issues/16)
[Issue 17](https://github.com/vgvassilev/clad/issues/17)
[Issue 18](https://github.com/vgvassilev/clad/issues/18)
[Issue 19](https://github.com/vgvassilev/clad/issues/19)
[Issue 20](https://github.com/vgvassilev/clad/issues/20)
[Issue 21](https://github.com/vgvassilev/clad/issues/21)
[Issue 23](https://github.com/vgvassilev/clad/issues/23)
[Issue 24](https://github.com/vgvassilev/clad/issues/24)
[Issue 25](https://github.com/vgvassilev/clad/issues/25)
[Issue 26](https://github.com/vgvassilev/clad/issues/26)
[Issue 27](https://github.com/vgvassilev/clad/issues/27)
[Issue 29](https://github.com/vgvassilev/clad/issues/29)
[Issue 40](https://github.com/vgvassilev/clad/issues/40)
[Issue 61](https://github.com/vgvassilev/clad/issues/61)

<!---Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' --->
<!---Get release bugs
git log v0.1..master | grep 'Fixes' | \
  s,^.*([0-9]+).*$,[\1]\(https://github.com/vgvassilev/clad/issues/\1\),' | uniq
--->
<!---Standard MarkDown doesn't support neither variables nor <base>
[Issue XXX](https://github.com/vgvassilev/clad/issues/XXX)
--->


Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

Vassil Vassilev (214)
Martin Vassilev (108)
Alexander Penev (25)
Violeta Ilieva (19)
Aleksandr Efremov (10)
Oksana Shadura (2)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.1...master | sort | uniq -c | sort -rn
--->
