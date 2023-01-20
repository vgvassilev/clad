Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 1.1. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 1.1?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-5.0 to clang-15


Forward Mode
------------
* Fix a bug in pow pushforward


Reverse Mode
------------
* Improve for-loop conditions


Error Estimation
----------------
* Improvements in error estimation of arrays
* Add error estimation example


Fixed Bugs
----------

[430](https://github.com/vgvassilev/clad/issues/430)
[474](https://github.com/vgvassilev/clad/issues/474)
[505](https://github.com/vgvassilev/clad/issues/505)
[506](https://github.com/vgvassilev/clad/issues/506)
[507](https://github.com/vgvassilev/clad/issues/507)
[514](https://github.com/vgvassilev/clad/issues/514)
[515](https://github.com/vgvassilev/clad/issues/515)

 <!---Get release bugs
 git log v1.0..master | grep 'Fixes|Closes'
 --->

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

FirstName LastName (#commits)

A B (N)

Alexander Penev (5)
Vassil Vassilev (3)
vidushi (2)
ioanaif (2)
Vaibhav Thakkar (1)
Parth Arora (1)
Garima Singh (1)
Baidyanath Kundu (1)

<!---Find contributor list for this release
 git log --pretty=format:"%an"  v1.0...master | sort | uniq -c | sort -rn |\
   sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
