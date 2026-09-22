// An analysis Clad's Analyses does not list, so nothing would render it: no
// entry in the table, no switch, no line on the help screen.
include "Analyses.td"

def Unlisted : Analysis<"unlisted", "unlisted", 0, "this must never render">;
