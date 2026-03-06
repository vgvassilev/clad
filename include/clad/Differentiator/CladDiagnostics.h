#ifndef CLAD_DIAGNOSTICS_H
#define CLAD_DIAGNOSTICS_H

#include "clang/Basic/Diagnostic.h"

namespace clad {

enum class DiagnosticGroup {
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

#endif
