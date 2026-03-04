#ifndef CLAD_DIAGNOSTICS_H
#define CLAD_DIAGNOSTICS_H

namespace clad {

enum class DiagnosticGroup {
  Clad,
  CladUnsupported,
  CladCheckpointing,
  CladPragma,
  CladBuiltin,
  CladNonDifferentiable,
};

} // namespace clad

#endif
