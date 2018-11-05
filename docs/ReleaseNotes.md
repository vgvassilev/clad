Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.4. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Cla in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.4?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------
* clang-5.0

Forward Mode
------------
* Support `x += y`, `x -= y`, `x *= y`, `x /= y`, `x++`, `x--`, `++x`, `--x`.
* Reduce emission of unused expressions -- in a few (trivial) cases we know if
  the expression is unused and clang will produce a warning. We use the same
  heuristics to detect such cases and produce less code.

Reverse Mode
------------
* Improve integration with CERN's data analysis framework [ROOT](https://github.com/root-project/root).

Misc
----
* Generate the derivatives in the correct namespace.
* Improve error handling in the cases we failed to produce a derivative.
* Reduce amount of clang libraries we link against.
* Exclude code which clad sees but knows has no derivatives.
* Add a special `#pragma clad ON/OFF/DEFAULT` to annotate regions which contain
  derivatives.

Fixed Bugs
----------

[Issue 97](https://github.com/vgvassilev/clad/issues/97)

<!---Uniquify by sort ReleaseNotes.md | uniq -c | grep -v '1 ' --->
<!---Get release bugs
git log v0.3..master | grep 'Fixes' | \
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
Aleksandr Efremov (4)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.3...master | sort | uniq -c | sort -rn
--->
