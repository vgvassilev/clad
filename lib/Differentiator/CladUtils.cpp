#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/Decl.h"

namespace clad {
  namespace utils {
    std::string ComputeEffectiveFnName(const clang::FunctionDecl* FD) {
      // TODO: Add cases for more operators
      switch (FD->getOverloadedOperator()) {
        case clang::OverloadedOperatorKind::OO_Call: return "operator_call";
        default: return FD->getNameAsString();
      }
    }
  } // namespace utils
} // namespace clad