#include "clad/Differentiator/DerivedFnInfo.h"
#include "clad/Differentiator/DiffPlanner.h"

using namespace clang;

namespace clad {
DerivedFnInfo::DerivedFnInfo(const DiffRequest& request,
                             FunctionDecl* derivedFn,
                             FunctionDecl* overloadedDerivedFn)
    : m_OriginalFn(request.Function), m_DerivedFn(derivedFn),
      m_OverloadedDerivedFn(overloadedDerivedFn), m_Mode(request.Mode),
      m_DiffVarsInfo(request.DVI), m_UsesEnzyme(request.use_enzyme),
      m_DeclarationOnly(request.DeclarationOnly) {}

bool DerivedFnInfo::SatisfiesRequest(const DiffRequest& request) const {
  return (request.Function == m_OriginalFn && request.Mode == m_Mode &&
          request.DVI == m_DiffVarsInfo && request.use_enzyme == m_UsesEnzyme &&
          request.DeclarationOnly == m_DeclarationOnly);
}

bool DerivedFnInfo::IsValid() const { return m_OriginalFn && m_DerivedFn; }

bool DerivedFnInfo::RepresentsSameDerivative(const DerivedFnInfo& lhs,
                                             const DerivedFnInfo& rhs) {
  return lhs.m_OriginalFn == rhs.m_OriginalFn && lhs.m_Mode == rhs.m_Mode &&
         lhs.m_DiffVarsInfo == rhs.m_DiffVarsInfo &&
         lhs.m_UsesEnzyme == rhs.m_UsesEnzyme &&
         lhs.m_DeclarationOnly == rhs.m_DeclarationOnly;
}
} // namespace clad
