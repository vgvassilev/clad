//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

/// Reading the command line, and the only place the option tables are
/// expanded. Everything below is one x-macro pass or another over Options.def
/// and Analyses.def; keeping it here is what stops that boilerplate reaching
/// the headers that merely want to know what was asked for.

#include "clad/Differentiator/Options.h"

#include "clad/Differentiator/Compatibility.h" // IWYU pragma: keep
#include "clad/Differentiator/Version.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <string>

namespace clad {

/// What the command line last said about one analysis. Unset means no switch
/// named it, so it runs at the default Analyses.td gives it. Not spelled
/// 'Default', which is a CLAD_ANALYSIS parameter and would be substituted
/// inside the macro bodies naming this enum.
enum class AnalysisSwitch : std::uint8_t { Unset, On, Off };

/// What the switches said while the command line was being read. Each analysis
/// takes its answer from the last switch that named it; the two bools record
/// which of the original -enable-X/-disable-X pair were seen, so that
/// giving both is diagnosed rather than resolved by order.
struct AnalysisFlags {
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  AnalysisSwitch Id##Switch = AnalysisSwitch::Unset;                           \
  bool Enable##Id##Analysis = false;                                           \
  bool Disable##Id##Analysis = false;
#include "clad/Differentiator/Analyses.def"

  /// Hands each analysis its answer: what a switch asked, or the default the
  /// table gives it.
  void resolveInto(Options& O) const {
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  O.Enable##Id##Analysis = (Id##Switch == AnalysisSwitch::Unset)               \
                               ? (Default)                                     \
                               : (Id##Switch == AnalysisSwitch::On);
#include "clad/Differentiator/Analyses.def"
  }

  /// Says where both halves of a pair were given, which is a contradiction
  /// rather than something to resolve by order.
  [[nodiscard]] bool contradicted() const {
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  if (Enable##Id##Analysis && Disable##Id##Analysis) {                         \
    llvm::errs() << "clad: Error: -enable-" Legacy " and -disable-" Legacy     \
                    " cannot be used together.\n";                             \
    return true;                                                               \
  }
#include "clad/Differentiator/Analyses.def"
    return false;
  }
};

/// Match one of the original per-analysis switches, -enable-X or
/// -disable-X. These are a switch per analysis rather than an option in
/// their own right, so Analyses.td spells them and Options.td does not.
static bool setAnalysisFromFlag(AnalysisFlags& F, llvm::StringRef Arg) {
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  if (Arg == "-enable-" Legacy) {                                              \
    F.Enable##Id##Analysis = true;                                             \
    F.Id##Switch = AnalysisSwitch::On;                                         \
    return true;                                                               \
  }                                                                            \
  if (Arg == "-disable-" Legacy) {                                             \
    F.Disable##Id##Analysis = true;                                            \
    F.Id##Switch = AnalysisSwitch::Off;                                        \
    return true;                                                               \
  }
#include "clad/Differentiator/Analyses.def"
  return false;
}

/// Says which analyses there are, where an option named one clad does not
/// have. \p Also names whatever else that option accepts.
static bool unknownAnalysis(llvm::StringRef Name, const char* Also) {
  llvm::errs() << "clad: Error: unknown analysis '" << Name << "'; known:";
#define CLAD_ANALYSIS(Id, AName, Legacy, Default, Desc)                        \
  llvm::errs() << " " AName;
#include "clad/Differentiator/Analyses.def"
  llvm::errs() << Also << "\n";
  return false;
}

/// Turns the named analysis on or off. The last switch naming it decides, so
/// a build that turns everything off by default can be overridden one
/// analysis at a time.
static bool setAnalysisByName(AnalysisFlags& F, llvm::StringRef Name,
                              AnalysisSwitch To) {
  // 'all' asks for the conservative derivative rather than for a particular
  // list, so it covers analyses added after the command line was written.
  // Only disabling has that meaning: falling back is safe for every analysis
  // by construction, while whether one is sound to run is what its default
  // encodes, so there is no configuration 'enable all' would name.
  if (Name == "all" && To == AnalysisSwitch::Off) {
#define CLAD_ANALYSIS(Id, AName, Legacy, Default, Desc)                        \
  F.Id##Switch = AnalysisSwitch::Off;
#include "clad/Differentiator/Analyses.def"
    return true;
  }

#define CLAD_ANALYSIS(Id, AName, Legacy, Default, Desc)                        \
  if (Name == (AName)) {                                                       \
    F.Id##Switch = To;                                                         \
    return true;                                                               \
  }
#include "clad/Differentiator/Analyses.def"
  return unknownAnalysis(Name,
                         /*Also=*/"; -fdisable-analysis also takes 'all'.");
}

/// Asks the named analysis for remarks about what it left in the generated
/// code.
static bool remarkAnalysisByName(Options& O, llvm::StringRef Name) {
#define CLAD_ANALYSIS(Id, AName, Legacy, Default, Desc)                        \
  if (Name == (AName)) {                                                       \
    O.Remark##Id##Analysis = true;                                             \
    return true;                                                               \
  }
#include "clad/Differentiator/Analyses.def"
  return unknownAnalysis(Name, /*Also=*/".");
}

/// Asks the named analysis to report what it concluded.
static bool dumpAnalysisByName(Options& O, llvm::StringRef Name) {
#define CLAD_ANALYSIS(Id, AName, Legacy, Default, Desc)                        \
  if (Name == (AName)) {                                                       \
    O.Dump##Id##Analysis = true;                                               \
    return true;                                                               \
  }
#include "clad/Differentiator/Analyses.def"
  return unknownAnalysis(Name, /*Also=*/".");
}

/// Which option an argument is. One per entry in Options.td.
enum class OptionID : std::uint8_t {
#define CLAD_OPTION(Id, Kind, Spelling, MetaVar, Help) Id,
#include "clad/Differentiator/Options.def"
  Unknown
};

/// A flag is the whole argument. A joined option is a prefix, and what
/// follows it is the value; the two are named so that an entry's Kind selects
/// one.
static bool matchesFlag(llvm::StringRef Arg, llvm::StringRef Spelling,
                        llvm::StringRef& Val) {
  Val = llvm::StringRef();
  return Arg == Spelling;
}
static bool matchesJoined(llvm::StringRef Arg, llvm::StringRef Spelling,
                          llvm::StringRef& Val) {
  if (!Arg.starts_with(Spelling))
    return false;
  Val = Arg.drop_front(Spelling.size());
  return true;
}

/// Which option \p Arg names, and for a joined one what followed its
/// spelling.
static OptionID matchOption(llvm::StringRef Arg, llvm::StringRef& Val) {
#define CLAD_OPTION(Id, Kind, Spelling, MetaVar, Help)                         \
  if (matches##Kind(Arg, Spelling, Val))                                       \
    return OptionID::Id;
#include "clad/Differentiator/Options.def"
  return OptionID::Unknown;
}

/// Writes what the option asked for into \p O, where the table says which
/// field it sets. False for an option that does something instead, which the
/// caller then has to act on.
static bool applyOption(Options& O, OptionID ID, llvm::StringRef Val) {
  // Only a joined option's value is read, and a table of nothing but
  // flags would leave this untouched.
  (void)Val;
  switch (ID) {
    // Only the options that set a field get a case; giving the rest one each
    // would be the same statement written out as many times as there are of
    // them.
#define CLAD_OPTION(Id, Kind, Spelling, MetaVar, Help)
#define CLAD_OPTION_WITH_MARSHALLING(Id, Kind, Spelling, MetaVar, Help, Type,  \
                                     KeyPath, Default, Value)                  \
  case OptionID::Id:                                                           \
    O.KeyPath = Value;                                                         \
    return true;
#include "clad/Differentiator/Options.def"
  default:
    return false;
  }
}

/// Every spelling clad accepts, in the table's order.
static constexpr std::array OptionSpellings = {
#define CLAD_OPTION(Id, Kind, Spelling, MetaVar, Help) Spelling,
#include "clad/Differentiator/Options.def"
};

/// The spelling nearest \p Arg, or empty where nothing is near enough to be
/// worth naming. Measured as clang's OptTable::findNearest measures it, and
/// offered on the same terms its driver offers one: within a single edit,
/// past which a suggestion misleads more often than it helps.
static llvm::StringRef nearestOption(llvm::StringRef Arg) {
  // A joined option's value is not part of its spelling, so a misspelt
  // -Rclad-analysis=loop is measured against -Rclad-analysis= rather than
  // against the whole argument, which the value would dominate.
  llvm::StringRef Head = Arg;
  if (auto Eq = Arg.find('='); Eq != llvm::StringRef::npos)
    Head = Arg.take_front(Eq + 1);

  llvm::StringRef Best;
  unsigned BestDistance = UINT_MAX;
  for (const char* S : OptionSpellings) {
    llvm::StringRef Spelling = S;
    unsigned Distance = std::min(
        Arg.edit_distance(Spelling, /*AllowReplacements=*/true, BestDistance),
        Head.edit_distance(Spelling, /*AllowReplacements=*/true, BestDistance));
    if (Distance < BestDistance) {
      BestDistance = Distance;
      Best = Spelling;
    }
  }
  return BestDistance <= 1 ? Best : llvm::StringRef();
}

/// The help screen, read out of the tables that define what it describes:
/// every option from Options.td, then the per-analysis switches from
/// Analyses.td, which is a pair per analysis rather than an option of its own.
static void printHelp() {
  llvm::errs() << "Options specific to Clad (preceded by -plugin-arg-clad):\n";
#define CLAD_OPTION(Id, Kind, Spelling, MetaVar, Help)                         \
  llvm::errs() << Spelling MetaVar " - " Help "\n";
#include "clad/Differentiator/Options.def"

  llvm::errs()
      << "Each analysis below is optional: it changes the code clad "
         "generates, never the values that code computes. Enabling one asks "
         "clad to prove more and store less; disabling one falls back to the "
         "conservative derivative.\n";
#define CLAD_ANALYSIS(Id, AName, Legacy, Default, Desc)                        \
  llvm::errs() << "-enable-" Legacy " / -disable-" Legacy                      \
                  " - Turns the " AName                                        \
                  " analysis on or off for the whole translation unit, "       \
                  "unless an individual request specifies otherwise. It " Desc \
                  ". Default: "                                                \
               << ((Default) ? "on" : "off") << ".\n";
#include "clad/Differentiator/Analyses.def"
  llvm::errs() << "\n";
}

bool Options::read(llvm::ArrayRef<std::string> Args) {
  AnalysisFlags Switches;
  for (const std::string& A : Args) {
    llvm::StringRef Arg = A;
    llvm::StringRef Val;
    OptionID ID = matchOption(Arg, Val);
    // Most options only set a field, and Options.td says which.
    if (applyOption(*this, ID, Val))
      continue;
    switch (ID) {
    case OptionID::Rclad_analysis_EQ:
      if (!remarkAnalysisByName(*this, Val))
        return false;
      break;
    case OptionID::fdump_analysis_EQ:
      if (!dumpAnalysisByName(*this, Val))
        return false;
      break;
    case OptionID::fenable_analysis_EQ:
      if (!setAnalysisByName(Switches, Val, AnalysisSwitch::On))
        return false;
      break;
    case OptionID::fdisable_analysis_EQ:
      if (!setAnalysisByName(Switches, Val, AnalysisSwitch::Off))
        return false;
      break;
    case OptionID::fcustom_estimation_model:
      llvm::errs() << "`-fcustom-estimation-model` is deprecated.\n";
      return false;
    case OptionID::help:
      // The frontend's own ShowHelp does not give us control.
      printHelp();
      break;
    case OptionID::version:
    case OptionID::v:
      llvm::errs() << getCladFullVersion() << "\n";
      break;
    case OptionID::Unknown:
      if (setAnalysisFromFlag(Switches, Arg))
        break;
      llvm::errs() << "clad: Error: invalid option " << Arg;
      if (llvm::StringRef Near = nearestOption(Arg); !Near.empty())
        llvm::errs() << "; did you mean " << Near << "?";
      llvm::errs() << "\n";
      return false;
    default:
      break;
    }
  }

  if (Switches.contradicted())
    return false;
  // Every switch has been read, so each analysis can be told what it was
  // asked for; nothing downstream has to know which switch said so.
  Switches.resolveInto(*this);
  return true;
}

} // namespace clad
