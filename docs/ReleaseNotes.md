Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 0.3. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 0.3?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------
* clang-5.0

Misc
----

* Improvements in both Forward and Reverse mode:
  * Better correctness of C++ constructs -- handle scopes properly; allow proper
  variable shadowing; and preserve namespaces.

* Forward Mode:
  * Efficient evaluation in forward mode -- for given:
  ```cpp
  double t = std::sin(x) * std::cos(x);
  ```
  is converted into:
  ```cpp
  double _t0 = std::sin(x);
  double _t1 = std::cos(x);
  double _d_t = sin_darg0(x) * (_d_x) * _t1 + _t0 * cos_darg0(x) * (_d_x);
  double t = _t0 * _t1;
  ```
  instead of
  ```cpp
  double _d_t = sin_darg0(x) * (_d_x) * cos(x) + sin(x) * cos_darg0(x) * (_d_x);
  double t = sin(x) * cos(x);
  ```
  to avoid re-evaluation.

  * Reduced cloning complexity -- the recursive cloning of complexity O(n^2) is
  replaced by cloning the whole tree at once yielding O(2n).

  * Handle more C++ constructs -- variable reassignments and for loops.



Fixed Bugs
----------

[Issue 47](https://github.com/vgvassilev/clad/issues/47)
[Issue 92](https://github.com/vgvassilev/clad/issues/92)

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

Aleksandr Efremov (6)
Vassil Vassilev (2)

<!---Find contributor list for this release
git log --pretty=format:"%an"  v0.2...master | sort | uniq -c | sort -rn
--->
