Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.2. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Cla in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.2?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

Misc
----
* Support Windows -- clad can compile on Windows.
* Improve build infrastructure.

Fixed Bugs
----------

[Issue 78](https://github.com/vgvassilev/clad/issues/78)
[Issue 74](https://github.com/vgvassilev/clad/issues/74)

<!---Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' --->
<!---Get release bugs
git log v0.2..master | grep 'Fixes' | \
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

Vassil Vassilev (7)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.2...master | sort | uniq -c | sort -rn
--->
