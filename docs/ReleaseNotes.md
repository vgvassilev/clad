Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 2.4. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 2.4?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-12 to clang-22. Support for clang-11 has been
  dropped as the support window shifted to the last ten major releases.


Forward Mode & Reverse Mode
---------------------------
* Add support for the OpenMP `critical` directive in both forward and reverse
  mode.
* Safely fall back when differentiating functions with no parameters instead of
  crashing.

Forward Mode
------------
*

Reverse Mode
------------
* Handle member access on call results by seeding the corresponding member of
  the record adjoint before visiting the call, so expressions like
  `objVal(x).val` emit the needed pullback call instead of an empty gradient.
* Allow redefining builtin custom derivatives: lookup now prioritizes
  `custom_derivatives::overrides` and only falls back to the builtin when no
  overrides declaration exists.

CUDA
----
* Extend tape functionality to support GPU offloading to VRAM.
* Add `CUDA_HOST_DEVICE` to `Tape::destroy_element()` so it can be called from
  device code.

Error Estimation
----------------
*

Misc
----
* Guard statement scopes with a `ScopeRAII` handle so scope balance is enforced
  by the type rather than by hand (NFC).
* Skip backend pass registration when clad is built as a static library
  (`CLAD_BUILD_STATIC_ONLY=ON`), fixing rootcling dictionary generation in ROOT.
* Use `std::unique_ptr` for `ReverseModeVisitor::m_ExternalSource` to make
  ownership explicit and exception-safe.
* Add LLVM 22 AddressSanitizer CI coverage and resolve the issues it surfaced.

Fixed Bugs
----------

[767](https://github.com/vgvassilev/clad/issues/767)
[1298](https://github.com/vgvassilev/clad/issues/1298)
[1376](https://github.com/vgvassilev/clad/issues/1376)
[1624](https://github.com/vgvassilev/clad/issues/1624)
[1815](https://github.com/vgvassilev/clad/issues/1815)
[1839](https://github.com/vgvassilev/clad/issues/1839)
[1855](https://github.com/vgvassilev/clad/issues/1855)

 <!---Get release bugs. Check for close, fix, resolve
 git log v2.3..master | grep -i "close" | grep '#' | sed -E 's,.*\#([0-9]*).*,\[\1\]\(https://github.com/vgvassilev/clad/issues/\1\),g' | sort -t'[' -k2,2n
 --->

<!--- https://github.com/vgvassilev/clad/issues?q=is%3Aissue%20state%3Aclosed%20closed%3A%3E2025-10-01 --->

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

Vassil Vassilev (12)
Vedant2005goyal (5)
Md Saif Ali Molla (2)
dependabot[bot] (2)
Shubham Shukla (1)
SahilPatidar (1)
leetcodez (1)
Devajith Valaparambil Sreeramaswamy (1)

<!---Find contributor list for this release
 git log --pretty=format:"%an"  v2.3...master | sort | uniq -c | sort -rn | sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
