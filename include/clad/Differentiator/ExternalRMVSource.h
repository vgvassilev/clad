#ifndef CLAD_EXTERNAL_RMV_SOURCE_H
#define CLAD_EXTERNAL_RMV_SOURCE_H

#include "llvm/ADT/SmallVector.h"

namespace clang {
  class ValueDecl;
}

namespace clad {

  class ReverseModeVisitor;
  class DiffRequest;
  // Should we include `DerivativeBuilder.h` instead? `DiffParams` is originally
  // defined in `DerivativeBuilder.h`.
  using DiffParams = llvm::SmallVector<const clang::ValueDecl*, 16>;

  /// An abstract interface that should be implemented by external sources
  /// that provide additional behaviour, in the form of callbacks at crucial
  /// locations, to the reverse mode visitor.
  class ExternalRMVSource {
  public:
    ExternalRMVSource() = default;

    /// Initialise the external source with the ReverseModeVisitor
    virtual void InitialiseRMV(ReverseModeVisitor& RMV) {}

    /// Informs the external source that associated `ReverseModeVisitor`
    /// object is no longer available.
    virtual void ForgetRMV() {};

    /// This is called at the beginning of the `ReverseModeVisitor::Derive`
    /// function.
    virtual void ActOnStartOfDerive() {}

    /// This is called at the end of the `ReverseModeVisitor::Derive`
    /// function.
    virtual void ActOnEndOfDerive() {}

    /// This is called just after differentiation arguments are parsed
    /// in `ReverseModeVisitor::Derive`.
    ///
    ///\param[in] request differentiation request
    ///\param[in] args differentiation args
    virtual void ActAfterParsingDiffArgs(const DiffRequest& request,
                                         DiffParams& args) {}

    /// This is called just before the scopes are created for the derived
    /// function.
    virtual void ActBeforeCreatingDerivedFnScope() {}

    /// This is called just after the scopes for the derived functions have been
    /// created.
    virtual void ActAfterCreatingDerivedFnScope() {}

    /// This is called at the beginning of the derived function body.
    virtual void ActOnStartOfDerivedFnBody() {}

    /// This is called at the end of the derived function body.
    virtual void ActOnEndOfDerivedFnBody() {}
  };
} // namespace clad
#endif