
#include "clad/Differentiator/DiffPlanner.h"

namespace clad {
  namespace internal {
    void symbol_requester() {
      DiffPlan plan;
      plan.dump();
    }
  }
}
