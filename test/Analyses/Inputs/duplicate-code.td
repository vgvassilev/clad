// A construct reusing a code another one already owns. Not listed in Clad's
// Descs, because the check looks at every construct the file defines: a
// retired one still owns its number.
include "Analyses.td"

def DuplicateOfCountedLoop : Desc<Loop, 1001, "a duplicate",
                                  "this must never render">;
