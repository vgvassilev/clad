#!/usr/bin/env python3
"""Check that the tests a change adds can actually fail without it.

A test that still passes with the change's functional hunks reverted is not
evidence about that change. Coverage does not show this, because every clad
test drags the whole pipeline through: on a sample batch seven of eight
tests executed a one-line patch only two of them could observe.

The caller reverts and rebuilds; this runs the touched test files against
the result and records, per test, whether it ran at all and whether it
still passes. A test this platform does not support is recorded as not
run, since a row that never executed one learned nothing about it. A
failure to build is reported apart from a FileCheck mismatch, since it
usually means the test needs a new API rather than that it pins a
behavior.

Exits 0 either way. One platform cannot decide the question -- a defect can
be invisible everywhere but under valgrind -- so it writes a verdict and
observes-change-verdict.py combines them.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# A clang diagnostic and a FileCheck diagnostic share this shape; only the
# message tells them apart.
DIAG = re.compile(r"^[^\s].*?:\d+:\d+: error: (.*)$", re.M)
# lit exits 0 for a test it never ran: an unmet REQUIRES: is UNSUPPORTED,
# not a failure. Only --show-unsupported puts that in the output, and
# without it a test nothing executed is indistinguishable from one that
# ran and passed.
UNSUPPORTED = re.compile(r"^UNSUPPORTED: ", re.M)
# Project convention, not anything intrinsic; each is overridable so a
# project laid out differently configures rather than forks.
TEST_DIRS = ("test",)
TEST_SUFFIXES = (".C", ".cpp", ".cu")
FUNCTIONAL_DIRS = ("lib", "include", "tools")
FUNCTIONAL_SUFFIXES = (".h", ".cpp", ".inc")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=True).stdout.split()


def changed(base_rev, test_dirs, test_suffixes, func_dirs, func_suffixes):
    # --diff-filter=d drops deletions: a test the change removes, or the old
    # path of a renamed one, no longer exists to run, and handing it to lit
    # would fail and be miscounted as evidence.
    files = git("diff", "--name-only", "--diff-filter=d", f"{base_rev}...HEAD")
    tests = [f for f in files
             if f.startswith(tuple(d + "/" for d in test_dirs))
             and Path(f).suffix in test_suffixes and Path(f).exists()]
    functional = [f for f in files
                  if f.startswith(tuple(d + "/" for d in func_dirs))
                  and Path(f).suffix in func_suffixes]
    return tests, functional


def clad_obj_root(build: Path):
    """Use the value lit was configured with rather than guessing paths."""
    cfg = build / "test" / "lit.site.cfg"
    if not cfg.exists():
        sys.exit(f"{cfg} not found -- is {build} a configured clad build?")
    m = re.search(r'^config\.clad_obj_root = "(.*)"$', cfg.read_text(), re.M)
    if not m:
        sys.exit(f"clad_obj_root not set in {cfg}")
    return Path(m.group(1))


def annotate(level, test, message):
    """Report this row's own result while the rest of the suite still runs.

    The aggregate cannot decide before every platform reports, but one
    row's result is known much earlier and belongs on the test file in the
    pull request rather than at the bottom of a log.
    """
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::{level} file={test},line=1::{message}")


def classify(output):
    diags = DIAG.findall(output)
    if any(not d.startswith("CHECK") for d in diags):
        return "COMPILE", "does not build at baseline -- needs a new API"
    if diags:
        return "CHECK", "clad generates different code at baseline"
    return "FAIL", "fails at baseline"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", help="configured clad build directory")
    ap.add_argument("--base-rev", default="origin/master",
                    help="revision the change is measured against")
    ap.add_argument("--platform", default="local",
                    help="name of the CI row this verdict comes from")
    ap.add_argument("--verdict", help="write the verdict as JSON here")
    ap.add_argument("--lit", default=shutil.which("lit"))
    ap.add_argument("--list", action="store_true",
                    help="print the touched test files and exit")
    ap.add_argument("--test-dirs", nargs="+", default=list(TEST_DIRS))
    ap.add_argument("--test-suffixes", nargs="+", default=list(TEST_SUFFIXES))
    ap.add_argument("--functional-dirs", nargs="+",
                    default=list(FUNCTIONAL_DIRS))
    ap.add_argument("--functional-suffixes", nargs="+",
                    default=list(FUNCTIONAL_SUFFIXES))
    ap.add_argument("tests", nargs="*",
                    help="test files to check (default: those the change touches)")
    a = ap.parse_args()

    tests, functional = ((a.tests, ["(explicit)"]) if a.tests
                         else changed(a.base_rev, a.test_dirs, a.test_suffixes,
                                      a.functional_dirs,
                                      a.functional_suffixes))
    # A change touching no functional code has nothing for its tests to
    # observe, and tests that pin existing behavior are expected to pass.
    if not functional:
        tests = []
    if a.list:
        print("\n".join(tests))
        return 0

    verdict = {"platform": a.platform, "base_sha": a.base_rev, "tests": {},
               "applicable": bool(tests)}
    if not tests:
        print("No touched lit tests to check.")
    else:
        if not a.build:
            sys.exit("--build is required unless --list is given")
        if not a.lit:
            sys.exit("lit not found; pass --lit or pip install lit")
        obj_root = clad_obj_root(Path(a.build))

        print(f"Running {len(tests)} touched test file(s) against clad built "
              f"from {a.base_rev}:\n")
        width = max(len(t) for t in tests)
        print(f"{'test':<{width + 2}}{'at baseline':<13}evidence")
        print("-" * (width + 55))
        for t in tests:
            # lit discovers tests through the build tree, so address them there.
            p = subprocess.run([a.lit, "-v", "--no-progress-bar",
                                "--show-unsupported", str(obj_root / t)],
                               capture_output=True, text=True)
            if p.returncode == 0:
                if UNSUPPORTED.search(p.stdout):
                    verdict["tests"][t] = {"ran": False, "passed": None,
                                           "kind": None}
                    print(f"{t:<{width + 2}}{'SKIP':<13}not run here -- "
                          f"REQUIRES not met")
                    annotate("notice", t, f"Not run on {a.platform}, so this "
                             "row has no evidence about it either way.")
                    continue
                verdict["tests"][t] = {"ran": True, "passed": True,
                                       "kind": None}
                print(f"{t:<{width + 2}}{'PASS':<13}none -- cannot fail for "
                      f"this change")
                annotate("warning", t, f"Passes on {a.platform} with this "
                         "change reverted, so it cannot be its regression "
                         "test. Another platform may still observe it.")
                continue
            kind, note = classify(p.stdout + p.stderr)
            verdict["tests"][t] = {"ran": True, "passed": False, "kind": kind}
            print(f"{t:<{width + 2}}{'FAIL':<13}{kind} -- {note}")
            annotate("notice", t, f"Observes this change on {a.platform} "
                     f"({kind}).")

    if a.verdict:
        Path(a.verdict).write_text(json.dumps(verdict, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
