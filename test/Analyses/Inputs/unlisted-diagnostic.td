// A message Clad's Diagnostics does not list, so nothing would render it.
include "Analyses.td"

def note_unlisted : Diagnostic<"Note", "this must never render">;
