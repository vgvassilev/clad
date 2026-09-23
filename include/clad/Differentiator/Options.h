//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

#ifndef CLAD_DIFFERENTIATOR_OPTIONS_H
#define CLAD_DIFFERENTIATOR_OPTIONS_H

#include "llvm/ADT/ArrayRef.h"

#include <optional>
#include <string>

namespace clad {

/// What the command line asked of clad.
///
/// One type for the whole way down: read() fills it, and a request reads it.
/// What each analysis is asked for is resolved while reading, so nothing
/// downstream has to know which switch said so.
///
/// The members come from Options.td, which is also where the spelling that
/// sets each one and the help that describes it are written. What any of them
/// means is documented there and in the generated reference page rather than
/// here, so that the two cannot disagree.
struct Options {
#define CLAD_OPTION(Id, Kind, Spelling, MetaVar, Help)
#define CLAD_OPTION_WITH_MARSHALLING(Id, Kind, Spelling, MetaVar, Help, Type,  \
                                     KeyPath, Default, Value)                  \
  Type KeyPath = Default;
#include "clad/Differentiator/Options.def"

  /// Whether each analysis runs, the switches already resolved against the
  /// default the table gives it. One member per entry in Analyses.def.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  bool Enable##Id##Analysis = Default;
#include "clad/Differentiator/Analyses.def"
  /// Whether the user asked to hear what each analysis left behind
  /// (-Rclad-analysis=NAME).
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  bool Remark##Id##Analysis = false;
#include "clad/Differentiator/Analyses.def"
  /// Whether -fdump-analysis=NAME asked for what each one concluded.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  bool Dump##Id##Analysis = false;
#include "clad/Differentiator/Analyses.def"

  /// Reads the arguments the plugin was given, in order. False where one of
  /// them was wrong, having already said so; the caller then declines to
  /// create the plugin.
  bool read(llvm::ArrayRef<std::string> Args);
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_OPTIONS_H
