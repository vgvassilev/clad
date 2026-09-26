// A construct Clad's Descs does not list, so nothing would render it: no entry
// in the table, no section on the page, no code for a report to print.
include "Analyses.td"

def UnlistedConstruct : Desc<Loop, 1099, "an unlisted construct",
                             "this must never render">;
