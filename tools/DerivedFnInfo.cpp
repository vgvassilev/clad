#include "DerivedFnInfo.h"

#include "clad/Differentiator/DiffPlanner.h"

using namespace clang;

namespace clad {
  DerivedFnInfo::DerivedFnInfo(const DiffRequest& request,
                               FunctionDecl* derivedFn,
                               FunctionDecl* overloadedDerivedFn)
      : m_OriginalFn(request.Function), m_DerivedFn(derivedFn),
        m_OverloadedDerivedFn(overloadedDerivedFn), m_Mode(request.Mode),
        m_DerivativeOrder(request.CurrentDerivativeOrder),
        m_DiffParamsInfo(request.DiffParamsInfo) {}

  bool DerivedFnInfo::SatisfiesRequest(const DiffRequest& request) const {
    return (request.Function == m_OriginalFn && request.Mode == m_Mode &&
            request.CurrentDerivativeOrder == m_DerivativeOrder &&
            request.DiffParamsInfo == m_DiffParamsInfo);
  }

  bool DerivedFnInfo::IsValid() const { return m_OriginalFn && m_DerivedFn; }

  bool DerivedFnInfo::RepresentsSameDerivative(const DerivedFnInfo& lhs,
                                               const DerivedFnInfo& rhs) {
    return lhs.m_OriginalFn == rhs.m_OriginalFn &&
           lhs.m_DerivativeOrder == rhs.m_DerivativeOrder &&
           lhs.m_Mode == rhs.m_Mode &&
           lhs.m_DiffParamsInfo == rhs.m_DiffParamsInfo;
  }
} // namespace clad