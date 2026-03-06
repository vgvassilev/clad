#include "clad/Differentiator/CladDiagnostics.h"

namespace clad {
namespace {
DiagnosticSettings& settings() {
  static DiagnosticSettings S;
  return S;
}
} // namespace

void resetDiagnosticSettings() { settings() = DiagnosticSettings{}; }

void setDiagnosticGroupEnabled(DiagnosticGroup Group, bool Enabled) {
  auto& S = settings();
  switch (Group) {
  case DiagnosticGroup::Clad:
    S.Clad = Enabled;
    break;
  case DiagnosticGroup::CladUnsupported:
    S.CladUnsupported = Enabled;
    break;
  case DiagnosticGroup::CladCheckpointing:
    S.CladCheckpointing = Enabled;
    break;
  case DiagnosticGroup::CladPragma:
    S.CladPragma = Enabled;
    break;
  case DiagnosticGroup::CladBuiltin:
    S.CladBuiltin = Enabled;
    break;
  case DiagnosticGroup::CladNonDifferentiable:
    S.CladNonDifferentiable = Enabled;
    break;
  }
}

bool isDiagnosticGroupEnabled(DiagnosticGroup Group) {
  const auto& S = settings();
  switch (Group) {
  case DiagnosticGroup::Clad:
    return S.Clad;
  case DiagnosticGroup::CladUnsupported:
    return S.CladUnsupported;
  case DiagnosticGroup::CladCheckpointing:
    return S.CladCheckpointing;
  case DiagnosticGroup::CladPragma:
    return S.CladPragma;
  case DiagnosticGroup::CladBuiltin:
    return S.CladBuiltin;
  case DiagnosticGroup::CladNonDifferentiable:
    return S.CladNonDifferentiable;
  }
  return true;
}

bool shouldEmitDiagnostic(DiagnosticGroup Group,
                          clang::DiagnosticsEngine::Level Level) {
  // Always preserve hard failures.
  if (Level == clang::DiagnosticsEngine::Error ||
      Level == clang::DiagnosticsEngine::Fatal) {
    return true;
  }

  if (!isDiagnosticGroupEnabled(DiagnosticGroup::Clad)) {
    return false;
  }

  return isDiagnosticGroupEnabled(Group);
}

} // namespace clad
