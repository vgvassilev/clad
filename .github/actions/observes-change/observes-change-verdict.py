#!/usr/bin/env python3
"""Combine what every platform reported into one verdict.

A test counts as observing the change if it failed at baseline on any row,
since a defect can be invisible everywhere but under valgrind. The check
fails only for a touched test that ran somewhere, failed nowhere, and whose
pre-change form passes with the change too. A row that skipped it (an unmet
REQUIRES:) reports no evidence; a test no row ran leaves the question open
rather than answered in the negative; and one whose pre-change form fails is
an edit this change forced, which belongs with it.

How many rows saw it is reported rather than collapsed: failing everywhere
is an ordinary regression test, while failing on exactly one is either the
most valuable kind of test in the suite or a flake, and a pass/fail bit
cannot tell a reviewer which.

The per-platform matrix is also written out. Accumulated across changes, a
test that is never the only one to fail is a candidate for merging -- a
claim about defects caught rather than lines executed.
"""

import json
import os
import sys
from pathlib import Path


def load(vdir):
    out = []
    for p in sorted(Path(vdir).rglob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (json.JSONDecodeError, OSError):
            print(f"skipping unreadable verdict {p}", file=sys.stderr)
    return out


def main(vdir, out=None):
    verdicts = load(vdir)
    if not verdicts:
        print("No verdicts reported -- nothing to decide.")
        return 0
    applicable = [v for v in verdicts if v.get("applicable")]
    if not applicable:
        print("No touched lit tests on any platform -- check not applicable.")
        return 0

    total = len(applicable)
    # test -> [(platform, kind)] where it failed at baseline, and
    # test -> [platform] for the rows that ran it at all.
    seen, ran, adapts, every = {}, {}, {}, set()
    for v in applicable:
        for name, r in v.get("tests", {}).items():
            every.add(name)
            # A row that skipped the test learned nothing about it. Older
            # verdicts predate the field and always ran what they report.
            if not r.get("ran", True):
                continue
            ran.setdefault(name, []).append(v["platform"])
            if not r.get("passed"):
                seen.setdefault(name, []).append((v["platform"], r.get("kind")))
            elif r.get("adapts"):
                adapts.setdefault(name, []).append(v["platform"])

    rows, narrow, blind, unrun, forced = [], [], [], [], []
    for name in sorted(every):
        where = seen.get(name, [])
        # Judge each test against the rows that ran it, not the rows that
        # reported: 5/5 among those that ran it is not 5/20 of the matrix.
        n, m, k = len(where), len(ran.get(name, [])), len(adapts.get(name, []))
        if m == 0:
            rows.append((name, f"not run on any of the {total}"))
            unrun.append(name)
        elif n == 0 and k:
            rows.append((name, f"adapts to the change on {k}/{m}"))
            forced.append(name)
        elif n == 0:
            rows.append((name, f"passes at baseline on all {m}"))
            blind.append(name)
        else:
            rows.append((name, f"observes the change on {n}/{m}"))
            # One row out of many is the interesting end of the range.
            if n == 1 and m > 1:
                narrow.append((name, where[0]))

    width = max(len(r[0]) for r in rows)
    lines = [f"Baseline pass reported by {total} platform(s), "
             f"{len(every)} touched test file(s).", ""]
    lines += [f"  {n:<{width + 2}}{s}" for n, s in rows]

    if narrow:
        lines += ["", f"Evidence from a single platform:", ""]
        for name, (plat, kind) in narrow:
            lines.append(f"  {name} -- only {plat} ({kind})")
        lines += ["", "That is the expected shape for a defect that only "
                  "manifests under one", "row's checking, such as valgrind or "
                  "a sanitizer. If this test is not", "about such a defect, "
                  "the result is more likely flaky than meaningful."]

    if forced:
        lines += ["", "These cannot fail without the change, but their "
                  "pre-change form fails with", "it -- edits the change "
                  "forced, which belong with it:", ""]
        lines += [f"  {f}" for f in forced]
        lines += ["", "Whether each adaptation is the right one is a "
                  "reviewer's call; that it is not", "an unrelated edit is "
                  "not."]

    if unrun:
        lines += ["", "No platform running this check executes these tests, "
                  "so it has no evidence", "about them either way:", ""]
        lines += [f"  {u}" for u in unrun]
        lines += ["", "The features their REQUIRES: names are absent from "
                  "every row that reports", "here. Either a row that has "
                  "them should run this check, or the change", "needs a test "
                  "the matrix can execute."]

    if blind:
        lines += ["", "No platform saw these fail with the change reverted, "
                  "so they cannot be", "its regression test:", ""]
        lines += [f"  {b}" for b in blind]
        lines += ["", "Either the shape they use does not reach the changed "
                  "code, or they pin", "behavior this change does not affect "
                  "-- in which case they belong in", "their own commit."]

    report = "\n".join(lines)
    print(report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        title = "Tests do not observe the change" if blind else \
                "Tests observe the change"
        md = [f"### {title}", "", "| test | baseline |", "| --- | --- |"]
        md += [f"| `{n}` | {s} |" for n, s in rows]
        if narrow:
            md += ["", "**Evidence from a single platform**", ""]
            md += [f"- `{n}` -- only `{p}` ({k})" for n, (p, k) in narrow]
        if forced:
            md += ["", "**Adaptations the change forced**", ""]
            md += [f"- `{f}`" for f in forced]
        if unrun:
            md += ["", "**Not run by any platform reporting here**", ""]
            md += [f"- `{u}`" for u in unrun]
        if blind:
            md += ["", "**No platform saw these fail with the change "
                   "reverted**", ""]
            md += [f"- `{b}`" for b in blind]
        with open(summary, "a") as f:
            f.write("\n".join(md) + "\n")

    if out:
        record = {
            "schema": 1,
            "pull_request": os.environ.get("PR_NUMBER"),
            "head_sha": os.environ.get("HEAD_SHA"),
            "base_sha": next((v.get("base_sha") for v in applicable
                              if v.get("base_sha")), None),
            "platforms": sorted(v["platform"] for v in applicable),
            "tests": {
                name: {
                    "observed_on": sorted(p for p, _ in seen.get(name, [])),
                    "kinds": sorted({k for _, k in seen.get(name, []) if k}),
                    "observed": len(seen.get(name, [])),
                    "ran_on": sorted(ran.get(name, [])),
                    "adapts_on": sorted(adapts.get(name, [])),
                    "of": len(ran.get(name, [])),
                }
                for name in sorted(every)
            },
            "blind": blind,
            "unrun": unrun,
            "forced": forced,
            "single_platform": [n for n, _ in narrow],
            "verdict": "blind" if blind else "observed",
        }
        Path(out).write_text(json.dumps(record, indent=2))
        print(f"\nWrote {out}")
    return 1 if blind else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a.split("=")[0]: a.split("=", 1)[-1]
             for a in sys.argv[1:] if a.startswith("--")}
    sys.exit(main(args[0], flags.get("--out")))
