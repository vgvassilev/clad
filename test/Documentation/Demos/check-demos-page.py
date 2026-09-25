#!/usr/bin/env python3
"""Keep the demos page, the demos' tests and demos/ telling the same story.

The page describes each demo and pulls the lines where it calls clad straight
out of the file, so the code on the page cannot drift. What can still drift is
which demos exist: add one and forget the page, or delete one and leave its
entry, and a reader is misled with nothing to catch it. That is what this
checks, along with every marker the page includes actually being there.

Each entry also links to its demo, and conf.py builds that link out of the
name, so a name that no longer matches the tree is a link into a 404. Checking
the name against the tree is therefore checking the link.

The tests under test/Demos are checked against the same list. Most of them
name their demo in a RUN line and would fail on their own if it moved, but the
CUDA ones are skipped wherever there is no toolkit, which is nearly
everywhere; reading their RUN lines here is what notices.
"""

import re
import sys
from pathlib import Path

# Nothing to list: shared headers, build files and data live beside the demos
# they belong to rather than being demos of their own.
NOT_A_DEMO = {"Makefile", "helper", "BlackScholes"}

MARKER = re.compile(r"docs-(?:begin|end)-[\w-]+")

# Two demos nothing here can build: cladtorch wants libtorch and something to
# train on, and the notebook wants a C++ Jupyter kernel. Neither is a
# dependency worth carrying to find out whether a demo still compiles.
UNCOMPILED = {"cladtorch", "Jupyter"}


def entries(page):
    """Demo names the page lists, as :demo:`Name` or :demo:`Dir/` entry titles."""
    text = Path(page).read_text()
    return set(re.findall(r"^:demo:`([A-Za-z][\w./-]*)`$", text, re.M))


def includes(page):
    """(file, marker) pairs the page pulls code from."""
    text = Path(page).read_text()
    return re.findall(
        r"literalinclude::\s*\S*?/demos/(\S+).*?:start-after:\s*(\S+)",
        text, re.S)


def stray_markers(root, page):
    """Markers clang-format has folded into the comment line above them.

    A marker sitting straight under a comment is part of that paragraph as far
    as clang-format is concerned, so re-wrapping an over-long line above it
    carries the marker along and leaves it inside a sentence. The page still
    renders, which is why this is worth saying out loud: separate the marker
    from the comment with a blank line.
    """
    problems = []
    for rel in sorted({rel for rel, _ in includes(page)}):
        f = Path(root) / "demos" / rel
        if not f.exists():
            continue
        for n, line in enumerate(f.read_text().splitlines(), 1):
            m = MARKER.search(line)
            if m and line.strip() != f"// {m.group()}":
                problems.append(f"demos/{rel}:{n} shares a line with "
                                f"{m.group()}; a marker needs one of its own")
    return problems


def compiled(root):
    """Demo paths the tests under test/Demos name in their RUN lines."""
    named = set()
    for t in sorted((Path(root) / "test" / "Demos").rglob("*")):
        if t.suffix in (".cpp", ".cu"):
            named |= set(re.findall(r"/demos/(\S+)", t.read_text()))
    return named


def demos(root):
    """Top-level entries under demos/, a directory counting as one demo."""
    return {p.name for p in sorted((Path(root) / "demos").iterdir())
            if not p.name.startswith(".") and p.name not in NOT_A_DEMO}


def main(root, page):
    listed, present = entries(page), demos(root)
    # A page entry may name a file inside a directory demo, so compare on the
    # top-level name in both directions.
    listed_top = {e.split("/")[0] for e in listed}
    problems = []
    for name in sorted(listed):
        path = Path(root) / "demos" / name.rstrip("/")
        if not path.exists():
            problems.append(f"the page links demos/{name}, which is missing")
        elif path.is_dir() != name.endswith("/"):
            kind = "a directory" if path.is_dir() else "a file"
            problems.append(f"demos/{name} is {kind}; the trailing slash marks "
                            "a directory and decides how the link is written")
    for missing in sorted(present - listed_top):
        problems.append(f"demos/{missing} exists but the page does not list it")
    for extra in sorted(listed_top - present):
        problems.append(f"the page lists {extra}, which is not in demos/")
    for rel, marker in includes(page):
        f = Path(root) / "demos" / rel
        if not f.exists():
            problems.append(f"the page includes demos/{rel}, which is missing")
        elif marker not in f.read_text():
            problems.append(f"demos/{rel} has no {marker} marker to include")
    problems += stray_markers(root, page)
    named = compiled(root)
    for demo in sorted(present - UNCOMPILED):
        if not any(n == demo or n.startswith(demo + "/") for n in named):
            problems.append(f"no test under test/Demos compiles demos/{demo}")
    for n in sorted(named):
        if not (Path(root) / "demos" / n).exists():
            problems.append(f"a test compiles demos/{n}, which is missing")

    for p in problems:
        print(f"  {p}")
    if problems:
        sys.exit(f"{len(problems)} disagreement(s) between the page and demos/")
    print(f"{len(present)} demos, all listed and all compiled but "
          f"{', '.join(sorted(UNCOMPILED))}; "
          f"{len(includes(page))} included excerpts, all present")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
