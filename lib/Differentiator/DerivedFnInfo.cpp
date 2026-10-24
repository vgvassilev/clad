#include "clad/Differentiator/DerivedFnInfo.h"
#include "clad/Differentiator/DiffRequest.h"

using namespace clang;

namespace clad {
DerivedFnInfo::DerivedFnInfo(const DiffRequest& request,
                             clang::FunctionDecl* derivedFn,
                             clang::FunctionDecl* overloadedDerivedFn)
    : m_OriginalFn(request.Function), m_DerivedFn(derivedFn),
      m_OverloadedDerivedFn(overloadedDerivedFn), m_Mode(request.Mode),
      m_DiffVarsInfo(request.DVI),
      m_CUDAGlobalArgsIndexes(request.CUDAGlobalArgsIndexes),
      m_UsesEnzyme(request.use_enzyme),
      m_DeclarationOnly(request.DeclarationOnly),
      m_FallbacksSeeded(request.FallbacksSeeded) {
#define CLAD_FALLBACK(Id, Opt, Flag, Default, FirstBit, Desc)                  \
  m_Enable##Id##Fallback = request.Enable##Id##Fallback;
#include "clad/Differentiator/Analyses.def"
}

bool DerivedFnInfo::SatisfiesRequest(const DiffRequest& request) const {
  return (request.Function == m_OriginalFn && request.Mode == m_Mode &&
          request.DVI == m_DiffVarsInfo && request.use_enzyme == m_UsesEnzyme &&
          request.DeclarationOnly == m_DeclarationOnly &&
          request.CUDAGlobalArgsIndexes == m_CUDAGlobalArgsIndexes);
}

const char* DerivedFnInfo::DisagreeingFallback(const DiffRequest& request,
                                               bool& Taken) const {
  if (!m_FallbacksSeeded || !request.FallbacksSeeded)
    return nullptr;
#define CLAD_FALLBACK(Id, Opt, Flag, Default, FirstBit, Desc)                  \
  if (m_Enable##Id##Fallback != request.Enable##Id##Fallback) {                \
    Taken = m_Enable##Id##Fallback;                                            \
    return Flag;                                                               \
  }
#include "clad/Differentiator/Analyses.def"
  return nullptr;
}

bool DerivedFnInfo::IsValid() const { return m_OriginalFn && m_DerivedFn; }

bool DerivedFnInfo::RepresentsSameDerivative(const DerivedFnInfo& lhs,
                                             const DerivedFnInfo& rhs) {
  return lhs.m_OriginalFn == rhs.m_OriginalFn && lhs.m_Mode == rhs.m_Mode &&
         lhs.m_DiffVarsInfo == rhs.m_DiffVarsInfo &&
         lhs.m_UsesEnzyme == rhs.m_UsesEnzyme &&
         lhs.m_DeclarationOnly == rhs.m_DeclarationOnly &&
         lhs.m_CUDAGlobalArgsIndexes == rhs.m_CUDAGlobalArgsIndexes;
}
} // namespace clad
