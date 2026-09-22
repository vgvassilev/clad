//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

#ifndef CLAD_DIFFERENTIATOR_DIAGNOSTICS_H
#define CLAD_DIFFERENTIATOR_DIAGNOSTICS_H

#include "clang/Basic/DiagnosticIDs.h"

#include "llvm/Support/ErrorHandling.h"

#include <cstdint>

namespace clad {

/// A message clad emits, one per entry in Analyses.td. The wording lives
/// there; a site names the message and fills in its placeholders.
///
/// Wider than the entries below need: only the messages the analyses emit are
/// in the table so far, and clad says far more than 256 things.
// NOLINTNEXTLINE(performance-enum-size)
enum class CladDiag : std::uint16_t {
#define CLAD_DIAG(Id, Severity, Text) Id,
#include "Diagnostics.def"
};

/// The message as the user reads it, clang's %0 placeholders included.
inline const char* textOf(CladDiag D) {
  switch (D) {
#define CLAD_DIAG(Id, Severity, Text)                                          \
  case CladDiag::Id:                                                           \
    return Text;
#include "Diagnostics.def"
  }
  llvm_unreachable("unhandled diagnostic"); // LCOV_EXCL_LINE
}

/// How loudly it is said. Spelled as DiagnosticIDs, which is what registers
/// a message whose text is not a literal.
inline clang::DiagnosticIDs::Level severityOf(CladDiag D) {
  switch (D) {
#define CLAD_DIAG(Id, Severity, Text)                                          \
  case CladDiag::Id:                                                           \
    return clang::DiagnosticIDs::Severity;
#include "Diagnostics.def"
  }
  llvm_unreachable("unhandled diagnostic"); // LCOV_EXCL_LINE
}

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_DIAGNOSTICS_H
