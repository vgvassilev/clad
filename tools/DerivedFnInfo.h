#ifndef CLAD_DERIVED_FN_H
#define CLAD_DERIVED_FN_H

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
    unsigned m_DerivativeOrder = 0;
    DiffInputVarsInfo m_DiffVarsInfo;
    bool m_UsesEnzyme = false;

    DerivedFnInfo() {}
    DerivedFnInfo(const DiffRequest& request, clang::FunctionDecl* derivedFn,
                  clang::FunctionDecl* overloadedDerivedFn);

    /// Returns true if the derived function represented by the object,
    /// satisfies the requirements of the given differentiation request.
    bool SatisfiesRequest(const DiffRequest& request) const;

    /// Returns true if the object represents any derived function; otherwise
    /// returns false.
    bool IsValid() const;

    const clang::FunctionDecl* OriginalFn() const { return m_OriginalFn; }
    clang::FunctionDecl* DerivedFn() const { return m_DerivedFn; }
    clang::FunctionDecl* OverloadedDerivedFn() const { return m_OverloadedDerivedFn; }

    /// Returns true if `lhs` and `rhs` represents same derivative.
    /// Here derivative is any function derived by clad.
    static bool RepresentsSameDerivative(const DerivedFnInfo& lhs,
                                         const DerivedFnInfo& rhs);
  };
} // namespace clad

#endif