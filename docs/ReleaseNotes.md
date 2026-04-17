Introduction
============

This document contains the release notes for the automatic differentiation
plugin for clang Clad, release 2.3. Clad is built on top of
[Clang](http://clang.llvm.org) and [LLVM](http://llvm.org>) compiler
infrastructure. Here we describe the status of Clad in some detail, including
major improvements from the previous release and new feature work.

Note that if you are reading this file from a git checkout,
this document applies to the *next* release, not the current one.


What's New in Clad 2.3?
========================

Some of the major new features and improvements to Clad are listed here. Generic
improvements to Clad as a whole or to its underlying infrastructure are
described first.

External Dependencies
---------------------

  * Clad now works with clang-11 to clang-21
  * Updated thid-party dependencies (e.g., `pygments`, `requests`, `urllib3`)
  * Improved CMake integration and external project configuration


Forward Mode & Reverse Mode
---------------------------

  * Expanded support for custom derivatives, including improved template deduction and implicit argument handling
  * Added derivatives for additional standard library functions (`std::beta`, `std::expint`, `std::comp_ellint`)
  * Improved handling of lambdas and nested differentiation requests
  * Better diagnostics and support for non-differentiable entities

Forward Mode
------------

  * Improved handling of const and non-differentiable parameters
  * More robust numerical differentiation constraints and diagnostics

Reverse Mode
------------

  * **Reduced overhead** by skipping unnecessary `reverse_forw` passes when custom pullbacks are available
  * Improved correctness and stability:
      * Fixed out-of-bounds access in nested Hessian computations
      * Safer adjoint initialization for complex record types
  * Improved scheduling to avoid unnecessary pullbacks
  * Added **basic OpenMP support**
  * Various fixes for pointer handling, checkpointing, and nested differentiation

CUDA
----

  * Improved CUDA support with compute-sanitizer integration
  * Fixed memory-related issues
  * Updated CI infrastructure for CUDA workflows

Error Estimation
----------------

  * Simplified and unified error estimation with reverse mode
  * Replaced custom models with a function-based interface
  * Improved correctness via direct error accumulation and numerical validation
  * Enabled more efficient handling of nested error estimation

Misc
----

  * Introduced a **dual-mode tape memory system**:
      * In-memory (RAM-disk via `mmap`) with automatic disk offloading
      * Support for user-defined custom tape implementations
  * Added new benchmarking capabilities
  * Improved CI performance and reliability
  * Numerous bug fixes and stability improvements across the codebase
  * Documentation updates, including guidance for non-differentiable annotations

Fixed Bugs
----------

[1808](https://github.com/vgvassilev/clad/issues/1808)
[1799](https://github.com/vgvassilev/clad/issues/1799)
[1793](https://github.com/vgvassilev/clad/issues/1793)
[1786](https://github.com/vgvassilev/clad/issues/1786)
[1761](https://github.com/vgvassilev/clad/issues/1761)
[1759](https://github.com/vgvassilev/clad/issues/1759)
[1750](https://github.com/vgvassilev/clad/issues/1750)
[1744](https://github.com/vgvassilev/clad/issues/1744)
[1739](https://github.com/vgvassilev/clad/issues/1739)
[1737](https://github.com/vgvassilev/clad/issues/1737)
[1736](https://github.com/vgvassilev/clad/issues/1736)
[1721](https://github.com/vgvassilev/clad/issues/1721)
[1713](https://github.com/vgvassilev/clad/issues/1713)
[1710](https://github.com/vgvassilev/clad/issues/1710)
[1701](https://github.com/vgvassilev/clad/issues/1701)
[1700](https://github.com/vgvassilev/clad/issues/1700)
[1699](https://github.com/vgvassilev/clad/issues/1699)
[1683](https://github.com/vgvassilev/clad/issues/1683)
[1682](https://github.com/vgvassilev/clad/issues/1682)
[1673](https://github.com/vgvassilev/clad/issues/1673)
[1654](https://github.com/vgvassilev/clad/issues/1654)
[1637](https://github.com/vgvassilev/clad/issues/1637)
[1629](https://github.com/vgvassilev/clad/issues/1629)
[1587](https://github.com/vgvassilev/clad/issues/1587)
[1577](https://github.com/vgvassilev/clad/issues/1577)
[1540](https://github.com/vgvassilev/clad/issues/1540)
[1526](https://github.com/vgvassilev/clad/issues/1526)
[1427](https://github.com/vgvassilev/clad/issues/1427)
[1420](https://github.com/vgvassilev/clad/issues/1420)
[1414](https://github.com/vgvassilev/clad/issues/1414)
[1363](https://github.com/vgvassilev/clad/issues/1363)
[1352](https://github.com/vgvassilev/clad/issues/1352)
[1350](https://github.com/vgvassilev/clad/issues/1350)
[1271](https://github.com/vgvassilev/clad/issues/1271)
[1043](https://github.com/vgvassilev/clad/issues/1043)
[1035](https://github.com/vgvassilev/clad/issues/1035)
[791](https://github.com/vgvassilev/clad/issues/791)
[627](https://github.com/vgvassilev/clad/issues/627)
[427](https://github.com/vgvassilev/clad/issues/427)

<!---Get release bugs. Check for close, fix, resolve
curl -s "https://api.github.com/search/issues?q=repo:vgvassilev/clad+is:issue+state:closed+closed:>2025-11-01&per_page=100" | jq -r '.items[] | "[\(.number)](\(.html_url))"'
--->

<!--- https://github.com/vgvassilev/clad/issues?q=is%3Aissue%20state%3Aclosed%20closed%3A%3E2025-10-01 --->

Special Kudos
=============

This release wouldn't have happened without the efforts of our contributors,
listed in the form of Firstname Lastname (#contributions):

Petro Zarytskyi (28; reverse mode improvements, error estimation refactoring, numerical differentiation constraints, diagnostics)
Vassil Vassilev (18; CI and infrastructure improvements, diagnostics, build system updates)
Vedant Goyal (12; tape memory system redesign, disk offloading, performance improvements, reverse mode optimizations)
utsav (4; template deduction fixes, CMake improvements, standard library derivatives)
ToYourLeft (4; Hessian correctness fixes, memory safety, pointer handling)
leetcodez (3; C++ standard support, memory handling fixes)
fogsong233 (3; CUDA fixes, memory safety improvements)
R Aadarsh (2; correctness fixes, ODR issues)
ovdiiuv (2; lambda rework and differentiation support)
mcbarton (2; dependency updates and CI improvements)
Fatima Essahbi (2; checkpointing fixes and reverse mode scheduling)
fatfat123-arch (2; checkpointing and reverse_forw fixes)
Shubham Shukla (1; co-contributions to Hessian and pointer handling fixes)
Rohan Timmaraju (1; CladTorch and GPT-2 integration work)
Maki Arima (1; documentation for custom derivatives)
Lucky Lodhi (1; benchmarking and comparisons)
Errant (1; OpenMP support in reverse mode)
Bertrand Bellenot (1; Windows-specific fixes)
Ayush Kumar (1; special function derivative support)
Aniruddha Adak (1; documentation for non-differentiable annotations)
Aditi Joshi (1; tape data structure improvements)
Aadeshveer Singh (1; CMake and Clang discovery improvements)

<!---Find contributor list for this release
 git log --pretty=format:"%an"  v2.2...master | sort | uniq -c | sort -rn | sed -E 's,^ *([0-9]+) (.*)$,\2 \(\1\),'
--->
