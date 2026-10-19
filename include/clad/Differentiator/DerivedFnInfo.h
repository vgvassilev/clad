#ifndef CLAD_DIFFERENTIATOR_DERIVEDFNINFO_H
#define CLAD_DIFFERENTIATOR_DERIVEDFNINFO_H

#include "clang/AST/Decl.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"

namespace clad {
struct DiffRequest;

/// `DerivedFnInfo` is designed to effectively store information about a
/// derived function.
struct DerivedFnInfo {
  const clang::FunctionDecl* m_OriginalFn = nullptr;
  clang::FunctionDecl* m_DerivedFn = nullptr;
  clang::FunctionDecl* m_OverloadedDerivedFn = nullptr;
  DiffMode m_Mode = DiffMode::unknown;
  DiffInputVarsInfo m_DiffVarsInfo;
  std::vector<size_t> m_CUDAGlobalArgsIndexes;
  bool m_UsesEnzyme = false;
  bool m_DeclarationOnly = false;
  /// Whether the derivative took each fallback. Not part of which request it
  /// satisfies: clad gives one function one derivative name, so a request
  /// differing only in a fallback is a conflict, not another derivative.
#define CLAD_FALLBACK(Id, Opt, Flag, Default, FirstBit, Desc)                  \
  bool m_Enable##Id##Fallback = Default;
#include "clad/Differentiator/Analyses.def"
  bool m_FallbacksSeeded = false;

  DerivedFnInfo() = default;
  DerivedFnInfo(const DiffRequest& request, clang::FunctionDecl* derivedFn,
                clang::FunctionDecl* overloadedDerivedFn);

  /// Returns true if the derived function represented by the object,
  /// satisfies the requirements of the given differentiation request.
  bool SatisfiesRequest(const DiffRequest& request) const;
  /// The fallback the derivative and \p request disagree on, as the command
  /// line spells it, or null where they agree or either was built only to be
  /// looked up. \p Taken says whether the derivative took it.
  const char* DisagreeingFallback(const DiffRequest& request,
                                  bool& Taken) const;

  /// Returns true if the object represents any derived function; otherwise
  /// returns false.
  bool IsValid() const;

  const clang::FunctionDecl* OriginalFn() const { return m_OriginalFn; }
  clang::FunctionDecl* DerivedFn() const { return m_DerivedFn; }
  clang::FunctionDecl* OverloadedDerivedFn() const {
    return m_OverloadedDerivedFn;
  }
  bool DeclarationOnly() const { return m_DeclarationOnly; }

  /// Returns true if `lhs` and `rhs` represents same derivative.
  /// Here derivative is any function derived by clad.
  static bool RepresentsSameDerivative(const DerivedFnInfo& lhs,
                                       const DerivedFnInfo& rhs);
};
} // namespace clad

#endif // CLAD_DIFFERENTIATOR_DERIVEDFNINFO_H
