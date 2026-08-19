#!/usr/bin/env python3
"""Combine what every platform reported into one verdict.

A test counts as observing the change if it failed at baseline on any row,
since a defect can be invisible everywhere but under valgrind. The check
fails only for a touched test that failed nowhere.

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
    # test -> [(platform, kind)] for the rows where it failed at baseline
    seen, every = {}, set()
    for v in applicable:
        for name, r in v.get("tests", {}).items():
            every.add(name)
            if not r.get("passed"):
                seen.setdefault(name, []).append((v["platform"], r.get("kind")))

    rows, narrow, blind = [], [], []
    for name in sorted(every):
        where = seen.get(name, [])
        n = len(where)
        if n == 0:
            rows.append((name, f"passes at baseline on all {total}"))
            blind.append(name)
        else:
            rows.append((name, f"observes the change on {n}/{total}"))
            # One row out of many is the interesting end of the range.
            if n == 1 and total > 1:
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
                    "of": total,
                }
                for name in sorted(every)
            },
            "blind": blind,
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
