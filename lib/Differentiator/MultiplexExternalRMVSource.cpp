#include "clad/Differentiator/MultiplexExternalRMVSource.h"
namespace clad {
  // void MultiplexExternalRMVSource::MultiplexExternalRMVSource() {}
  void MultiplexExternalRMVSource::AddSource(ExternalRMVSource& source) {
    m_Sources.push_back(&source);
  }

  void MultiplexExternalRMVSource::InitialiseRMV(ReverseModeVisitor& RMV) {
    for (auto source : m_Sources) {
      source->InitialiseRMV(RMV);
    }
  }

  void MultiplexExternalRMVSource::ForgetRMV() {
    for (auto source : m_Sources)
      source->ForgetRMV();
  }

  void MultiplexExternalRMVSource::ActOnStartOfDerive() {
    for (auto source : m_Sources) {
      source->ActOnStartOfDerive();
    }
  }

  void MultiplexExternalRMVSource::ActOnEndOfDerive() {
    for (auto source : m_Sources) {
      source->ActOnEndOfDerive();
    }
  }

  void MultiplexExternalRMVSource::ActAfterParsingDiffArgs(const DiffRequest& request,
                                                           DiffParams& args) {
    for (auto source : m_Sources) {
      source->ActAfterParsingDiffArgs(request, args);
    }
  }

  void MultiplexExternalRMVSource::ActBeforeCreatingDerivedFnScope() {
    for (auto source : m_Sources) {
      source->ActBeforeCreatingDerivedFnScope();
    }
  }

  void MultiplexExternalRMVSource::ActAfterCreatingDerivedFnScope() {
    for (auto source : m_Sources) {
      source->ActAfterCreatingDerivedFnScope();
    }
  }

  void MultiplexExternalRMVSource::ActOnStartOfDerivedFnBody() {
    for (auto source : m_Sources) {
      source->ActOnStartOfDerivedFnBody();
    }
  }

  void MultiplexExternalRMVSource::ActOnEndOfDerivedFnBody() {
    for (auto source : m_Sources) {
      source->ActOnEndOfDerivedFnBody();
    }
  }
} // namespace clad