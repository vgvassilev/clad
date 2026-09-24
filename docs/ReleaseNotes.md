Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 2.5. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 2.5?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

* Clad now works with clang-14 to clang-23. Support for clang-11 through
  clang-13 has been dropped: each was paying for itself in preprocessor
  branches whose first arm no supported configuration took.
* The Enzyme backend is bumped to v0.0.290 and is attached only where a
  request asks for it.


Forward Mode & Reverse Mode
---------------------------
* Generated code now carries real source locations. A debugger can step
  through the derivative clad wrote, and diagnostics about generated code
  point into it rather than at the request that asked for it;
  `-fgenerated-source-dir=<dir>` writes the generated code out for a debugger
  to read.
* `clad::immediate_mode` is gone. Clad decides for itself whether a derivative
  is needed while the program compiles, from where the call to
  `clad::differentiate` sits, so code passing the option should simply drop it.
  Every mode now gets it, which makes `clad::gradient` usable in an immediate
  context when compiled as C++26. `CladFunction` is a literal type.
* The analyses clad runs are described in one table, which drives the
  switches, the help screen and the reference page. Each can be turned on or
  off for a translation unit with `-fenable-analysis=`/`-fdisable-analysis=`,
  or for a single request with `clad::opts::enable_*`/`disable_*`;
  `-fdisable-analysis=all` asks for the conservative derivative throughout.
  The request options are named after the analyses
  (`clad::opts::enable_activity_analysis`, `enable_tbr_analysis`,
  `enable_useful_analysis`, `enable_loop_analysis`, each with a `disable_`
  twin); the short spellings (`enable_va`, `enable_ua`, ...) remain as aliases.
* `-Rclad-analysis=<name>` reports what an analysis left in the derivative and
  where, and says what it was looking for and did not find. Each construct an
  analysis looks for has a code (CLAD1001 and up) and a reference page.
* Clad's command-line options are described in one table too. A misspelled
  option is answered with the nearest spelling, and `-help` no longer
  advertises `-fcustom-estimation-model`, which is rejected as deprecated.
* `-fclad-porting-hints` names the custom derivatives a translation unit is
  missing, instead of clad silently descending into the library's internals.
* A counted loop's trip count is recomputed in the reverse sweep instead of
  being counted in the forward one, and the adjoint of a broadcast read -- an
  element read on every iteration at an index the loop never moves -- is
  summed in a register and reaches memory once after the loop.
* A callee records the ranges it wrote rather than tracking individual
  addresses.
* The activity analysis follows what a pointer writes through, not only the
  pointer, and the useful analysis runs only on functions with a body.
* `CLAD_NONDIFFERENTIABLE` marks types and members clad should not
  differentiate, and a type with an inaccessible copy constructor is treated
  as non-copyable.
* `assert` and other source-location builtins no longer stop differentiation.
* A type alias is carried into the derivative rather than refused, and a
  derivative that would read a variable before its declaration is diagnosed
  rather than emitted.
* A second `#pragma clad OFF` or `#pragma clad DEFAULT` is ignored rather
  than asserted on.
* `clad::zero_like` builds the zero adjoint of a value of any type clad can
  differentiate, and is what default adjoints are now built from.
* Only clad-internal derivatives are declared `inline`.

Forward Mode
------------
* A tangent known to be zero, including one bound to a variable or reached
  through pointer arithmetic, propagates as an identical zero rather than
  being fabricated.
* The constant folder runs in forward mode.
* Pushforwards for `lgamma`.

Reverse Mode
------------
* `#pragma omp parallel` regions are differentiated in reverse mode. A private
  variable's adjoint starts from zero in each thread, and a broadcast adjoint
  is summed per thread rather than into one shared address.
* Hessians are assembled from hessian-vector products, with sizes and
  diagnostics computed before deriving.
* Pullbacks for `std::vector` construction and `resize`, and zero pullbacks
  for `size()` and `capacity()`.
* Per-call state reaches a pullback from its `reverse_forw` through
  `pullback_state`, so a call inside a loop survives the replay.
* Pointers returned by `const` member functions get their adjoints.
* Reallocation is handled: a shrinking in-place `realloc` is undone in the
  reverse sweep rather than saved and restored around.
* Early returns are encoded with a named lambda, and a `switch` is reversed on
  its stored condition rather than on a second control-flow tape.

CUDA
----
* `threadIdx`, `blockIdx`, `blockDim` and `gridDim` are treated as passive, so
  neither the activity nor the to-be-recorded analysis spends an adjoint or a
  tape entry on them.
* `clad::restore_tracker` can be used in device kernels.
* Derivatives for more of Thrust, and the CUDA demos compile where there is no
  device to run them on.

Error Estimation
----------------
* No functional change. `clad::estimate_error` and what it computes are now
  documented in the user guide and the API reference.

Misc
----
* The demos are a directory worth browsing: one demo per capability, each
  compiled by the test suite, a helix fit among them. The Rosenbrock demo is
  absorbed by the Newton one, and the OpenCL demo is gone: it offloaded a
  Rosenbrock evaluation and differentiated nothing, so it showed a reader
  nothing about clad.
* The user guide is rewritten -- core concepts, reverse mode, what clad
  differentiates and how it declines, templates and overloads, a FAQ -- and
  its examples run as tests, so the documentation cannot drift from what clad
  does. The internal doxygen site is repaired, and a documentation mistake
  fails the build.
* The generated tables are rendered into the build directory by `clad-tblgen`
  rather than committed; clad builds inside an LLVM tree and against an LLVM
  build tree, not only against an installation.
* CI: an emscripten/wasm job, and the Basic unit tests build in cross-builds;
  clang-tidy and Valgrind run on pull requests; a gate checks that a change's
  tests fail without the change; clad runs inside cling after every merge.
* The LULESH and XSBench benchmarks, and a benchmark of the reverse-mode
  protocol against a replay-free one.
* The nix shell is replaced by an up-to-date flake.
* The Kokkos tests no longer include a non-public Kokkos header.

Fixed Bugs
----------

[357](https://github.com/vgvassilev/clad/issues/357)
[367](https://github.com/vgvassilev/clad/issues/367)
[373](https://github.com/vgvassilev/clad/issues/373)
[396](https://github.com/vgvassilev/clad/issues/396)
[403](https://github.com/vgvassilev/clad/issues/403)
[966](https://github.com/vgvassilev/clad/issues/966)
[1128](https://github.com/vgvassilev/clad/issues/1128)
[1145](https://github.com/vgvassilev/clad/issues/1145)
[1156](https://github.com/vgvassilev/clad/issues/1156)
[1181](https://github.com/vgvassilev/clad/issues/1181)
[1218](https://github.com/vgvassilev/clad/issues/1218)
[1265](https://github.com/vgvassilev/clad/issues/1265)
[1272](https://github.com/vgvassilev/clad/issues/1272)
[1275](https://github.com/vgvassilev/clad/issues/1275)
[1283](https://github.com/vgvassilev/clad/issues/1283)
[1409](https://github.com/vgvassilev/clad/issues/1409)
[1442](https://github.com/vgvassilev/clad/issues/1442)
[1446](https://github.com/vgvassilev/clad/issues/1446)
[1449](https://github.com/vgvassilev/clad/issues/1449)
[1571](https://github.com/vgvassilev/clad/issues/1571)
[1677](https://github.com/vgvassilev/clad/issues/1677)
[1693](https://github.com/vgvassilev/clad/issues/1693)
[1694](https://github.com/vgvassilev/clad/issues/1694)
[1804](https://github.com/vgvassilev/clad/issues/1804)
[1827](https://github.com/vgvassilev/clad/issues/1827)
[1855](https://github.com/vgvassilev/clad/issues/1855)
[1860](https://github.com/vgvassilev/clad/issues/1860)
[1865](https://github.com/vgvassilev/clad/issues/1865)
[1871](https://github.com/vgvassilev/clad/issues/1871)
[1872](https://github.com/vgvassilev/clad/issues/1872)
[1873](https://github.com/vgvassilev/clad/issues/1873)
[1916](https://github.com/vgvassilev/clad/issues/1916)
[1931](https://github.com/vgvassilev/clad/issues/1931)
[1940](https://github.com/vgvassilev/clad/issues/1940)
[1941](https://github.com/vgvassilev/clad/issues/1941)
[1947](https://github.com/vgvassilev/clad/issues/1947)
[1954](https://github.com/vgvassilev/clad/issues/1954)
[1955](https://github.com/vgvassilev/clad/issues/1955)
[1958](https://github.com/vgvassilev/clad/issues/1958)
[1960](https://github.com/vgvassilev/clad/issues/1960)
[2051](https://github.com/vgvassilev/clad/issues/2051)
[2083](https://github.com/vgvassilev/clad/issues/2083)
[2110](https://github.com/vgvassilev/clad/issues/2110)
[2111](https://github.com/vgvassilev/clad/issues/2111)
[2112](https://github.com/vgvassilev/clad/issues/2112)
[2113](https://github.com/vgvassilev/clad/issues/2113)
[2116](https://github.com/vgvassilev/clad/issues/2116)
[2174](https://github.com/vgvassilev/clad/issues/2174)
[2181](https://github.com/vgvassilev/clad/issues/2181)
[2194](https://github.com/vgvassilev/clad/issues/2194)

 <!---Get release bugs. Check for close, fix, resolve
 git log v2.4..master | grep -i "close" | grep '#' | sed -E 's,.*\#([0-9]*).*,\[\1\]\(https://github.com/vgvassilev/clad/issues/\1\),g' | sort -t'[' -k2,2n
 --->

<!--- https://github.com/vgvassilev/clad/issues?q=is%3Aissue%20state%3Aclosed%20closed%3A%3E2026-06-28
 gh api "search/issues?q=repo:vgvassilev/clad+is:issue+is:closed+closed:>=2026-06-28&per_page=100" --jq '.items[] | "\(.number) \(.state_reason)"' | sort -n
 --->

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

Vassil Vassilev (201)
Jonas Rembser (36)
Vedant Goyal (16)
Elvand Lie Nababan (7)
Shubham Shukla (5)
fogsong233 (5)
Aaron Jomy (4)
leetcodez (4)
Hardik Kumar (2)
Abdelrhman Elrawy (1)
Devajith Valaparambil Sreeramaswamy (1)
Matthew Barton (1)
Sahil Patidar (1)
Shresth Samyak (1)

<!---Find contributor list for this release
 git log --pretty=format:"%an"  v2.4...master | sort | uniq -c | sort -rn | sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
