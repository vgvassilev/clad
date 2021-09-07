#ifndef MULTIPLEX_EXTERNAL_RMV_SOURCE_H
#define MULTIPLEX_EXTERNAL_RMV_SOURCE_H

#include "llvm/ADT/SmallVector.h"
#include "clad/Differentiator/ExternalRMVSource.h"

namespace clad {
  class DiffRequest;

  // is `ExternalRMVSourceMultiplexer` a better name for the class?
  /// Manages multiple external RMV sources.
  class MultiplexExternalRMVSource : public ExternalRMVSource {
  private:
    llvm::SmallVector<ExternalRMVSource*, 4> m_Sources;

  public:
    MultiplexExternalRMVSource() = default;
    /// Adds `source` to the sequence of external RMV sources managed by this
    /// multiplexer.
    void AddSource(ExternalRMVSource& source);
    void InitialiseRMV(ReverseModeVisitor& RMV) override;
    void ForgetRMV() override;

    void ActOnStartOfDerive() override;
    void ActOnEndOfDerive() override;
    void ActAfterParsingDiffArgs(const DiffRequest& request,
                                 DiffParams& args) override;
    void ActBeforeCreatingDerivedFnScope() override;
    void ActAfterCreatingDerivedFnScope() override;
    void ActOnStartOfDerivedFnBody() override;
    void ActOnEndOfDerivedFnBody() override;
  };
} // namespace clad

#endif