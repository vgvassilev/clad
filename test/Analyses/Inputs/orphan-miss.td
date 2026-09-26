// A way to miss a construct that no construct lists, so nothing would render
// it: no note, no line on any page.
include "Analyses.td"

def OrphanMiss : Miss<"this must never render">;
