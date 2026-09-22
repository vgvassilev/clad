//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

#ifndef CLAD_DIFFERENTIATOR_OPTIONS_H
#define CLAD_DIFFERENTIATOR_OPTIONS_H

#include <string>

namespace clad {

/// What the command line asked of clad.
///
/// One type for the whole way down: the plugin fills it while parsing, and a
/// request reads it. What each analysis is asked for is resolved while
/// parsing, so nothing downstream has to know which switch said so.
struct Options {
  bool DumpSourceFn = false;
  bool DumpSourceFnAST = false;
  bool DumpDerivedFn = false;
  bool DumpDerivedAST = false;
  bool GenerateSourceFile = false;
  bool DumpGeneratedSource = false;
  /// Where to write the generated code so a debugger can open it, and
  /// whether that was asked at all. Asking with no directory writes nothing
  /// and says nothing, which is how a build that knows it does not want this
  /// turns the advice off.
  std::string GeneratedSourceDir;
  bool GeneratedSourceDirGiven = false;
  bool ValidateClangVersion = true;
  bool PrintNumDiffErrorInfo = false;
  bool EmitPortingHints = false;

  /// Whether each analysis runs, the switches already resolved against the
  /// default the table gives it. One member per entry in Analyses.def.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  bool Enable##Id##Analysis = Default;
#include "clad/Differentiator/Analyses.def"
  /// Whether the user asked to hear what each analysis left behind
  /// (-Rclad-analysis=\<name\>).
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  bool Remark##Id##Analysis = false;
#include "clad/Differentiator/Analyses.def"
  /// Whether -fdump-analysis=\<name\> asked for what each one concluded.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  bool Dump##Id##Analysis = false;
#include "clad/Differentiator/Analyses.def"
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_OPTIONS_H
