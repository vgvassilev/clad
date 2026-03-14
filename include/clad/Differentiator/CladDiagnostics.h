#ifndef CLAD_DIFFERENTIATOR_CLADDIAGNOSTICS_H
#define CLAD_DIFFERENTIATOR_CLADDIAGNOSTICS_H

#include "clang/Basic/Diagnostic.h"

#include <cstdint>

namespace clad {

enum class DiagnosticGroup : std::uint8_t {
  Clad,
  CladUnsupported,
  CladCheckpointing,
  CladPragma,
  CladBuiltin,
  CladNonDifferentiable,
};

struct DiagnosticSettings {
  bool Clad = true;
  bool CladUnsupported = true;
  bool CladCheckpointing = true;
  bool CladPragma = true;
  bool CladBuiltin = true;
  bool CladNonDifferentiable = true;
};

void resetDiagnosticSettings();
void setDiagnosticGroupEnabled(DiagnosticGroup Group, bool Enabled);
bool isDiagnosticGroupEnabled(DiagnosticGroup Group);
bool shouldEmitDiagnostic(DiagnosticGroup Group,
                          clang::DiagnosticsEngine::Level Level);

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_CLADDIAGNOSTICS_H
