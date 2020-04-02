Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.7. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.7?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------
* Clad now works with clang-5.0, clang-6.0, clang-7.0, clang-8.0 and clang-9.0

Forward Mode & Reverse Mode
---------------------------
*

Forward Mode
------------
*

Reverse Mode
------------
*

Misc
----
* 

Fixed Bugs
----------

[Issue XXX](https://github.com/vgvassilev/clad/issues/XXX)

<!---Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' --->
<!---Get release bugs
git log v0.6..master | grep 'Fixes' | \
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

* Author One (2)
* Author Two (1)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.6...master | sort | uniq -c | sort -rn
--->
