// An analysis whose pair of clad::opts bits runs into another analysis'.
// Each pair takes two adjacent bits, so 5 is the one the activity analysis
// already sits on.
include "Analyses.td"

def Overlapping : Analysis<"overlapping", "overlapping", 0, 5,
                           "this must never render">;
