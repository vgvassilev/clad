#ifndef CLAD_DIFFERENTIATOR_SCOPEEXIT_H
#define CLAD_DIFFERENTIATOR_SCOPEEXIT_H

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Config/llvm-config.h"

#include <utility>

namespace clad_compat {

// LLVM 22 exposes scope_exit with class template argument deduction and
// deprecates the old factory. Earlier supported LLVM releases expose the
// guard through that factory instead. Keep the version seam in one place:
// production cleanup must also compile when deprecations are errors.
template <typename Callable> [[nodiscard]] auto makeScopeExit(Callable&& F) {
#if LLVM_VERSION_MAJOR >= 22
  return llvm::scope_exit(std::forward<Callable>(F));
#else
  return llvm::make_scope_exit(std::forward<Callable>(F));
#endif
}

} // namespace clad_compat

#endif