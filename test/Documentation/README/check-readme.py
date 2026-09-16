"""Check that README.md's code blocks are the examples the test suite runs.

README.md cannot include a file the way the Sphinx pages can, so its examples
are copies. A copy drifts: every one of them was wrong before this check
existed. A test file opts a region into this check by carrying

    // docs-readme: <slug>

anywhere outside its regions, and this script then fails unless the region
between that slug's docs-begin/docs-end markers appears in README.md verbatim.
Discovering the pairs rather than listing them here means a new README example
is covered as soon as its test says so.
"""

import difflib
import os
import re
import sys


def region(path, slug):
    """The lines of `path` between the docs-begin/docs-end markers for `slug`."""
    text = open(path).read()
    begin = "// docs-begin-%s\n" % slug
    end = "// docs-end-%s" % slug
    if begin not in text or end not in text:
        sys.exit("%s: no docs-begin-%s/docs-end-%s markers" % (path, slug, slug))
    return text.split(begin)[1].split(end)[0].strip("\n")


def covered(root):
    """Every (test file, slug) pair that claims to appear in the README."""
    for dirpath, dirnames, names in os.walk(root):
        # Inputs holds fixtures for other tests, including one deliberately out
        # of step with a stand-in README; it is not documentation.
        dirnames[:] = [d for d in dirnames if d != "Inputs"]
        for name in sorted(names):
            if not name.endswith(".cpp"):
                continue
            path = os.path.join(dirpath, name)
            for slug in re.findall(r"^// docs-readme: (\S+)\s*$", open(path).read(),
                                   re.M):
                yield path, slug


def main():
    readme, tests = sys.argv[1], sys.argv[2]
    blocks = [b.strip("\n")
              for b in re.findall(r"```cpp\n(.*?)```", open(readme).read(), re.S)]

    pairs = sorted(covered(tests))
    if not pairs:
        sys.exit("no test claims a README region; expected at least one "
                 "'// docs-readme: <slug>' under %s" % tests)

    failed = False
    for path, slug in pairs:
        want = region(path, slug)
        if want in blocks:
            continue
        failed = True
        print("%s (%s) does not appear in %s as a ```cpp block."
              % (os.path.basename(path), slug, readme))
        near = max(blocks, key=lambda b: difflib.SequenceMatcher(None, b, want)
                   .ratio(), default=None)
        if near is not None:
            print("\n".join(difflib.unified_diff(
                near.split("\n"), want.split("\n"),
                fromfile="README.md", tofile=os.path.basename(path), lineterm="")))
    if failed:
        sys.exit("README.md and the tested examples have diverged. Copy the "
                 "region between the docs- markers into README.md, or change "
                 "the test and re-run it.")
    print("README.md matches %d tested example(s)." % len(pairs))


main()
