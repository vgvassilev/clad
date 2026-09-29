// An analysis whose position is below the bitmask's options. Counted from
// ORDER_BITS, so a negative one lands in the derivative order and a request
// asking for it would be read as asking for a different order.
include "Analyses.td"

def Negative : Analysis<"negative", "negative", 0, -5,
                        "this must never render">;
