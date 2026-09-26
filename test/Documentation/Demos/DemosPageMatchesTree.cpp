// The demos page pulls the lines where each demo calls clad out of the demo
// itself, so that code cannot drift. Which demos exist still can: this fails
// when one is added and not listed, listed and not present, or when the page
// includes a marker no longer in the file.
//
// RUN: %python %S/check-demos-page.py %S/../../.. %S/../../../docs/userDocs/source/user/demos.rst
