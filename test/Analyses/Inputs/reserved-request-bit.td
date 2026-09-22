// An analysis taking a bit one of clad::opts' hand-written options already
// owns. The two live in different files, which is why the table lists the
// reserved positions rather than leaving the collision to a static_assert.
include "Analyses.td"

def Reserving : Analysis<"reserving", "reserving", 0, 7,
                         "this must never render">;
