//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

/// Renders the tables under lib/Differentiator: the header clad compiles
/// against and the page its options are documented in. Everything
/// user-facing is written in a table; nothing here decides wording.
///
/// The header is committed, so building clad never runs this; regenerate it
/// with `ninja clad-options`, and test/Analyses/OptionTableUpToDate fails
/// when it no longer matches the table. The page is rendered where the
/// documentation is built.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

using namespace llvm;

// LLVM 20 made the RecordKeeper a TableGen main is handed const, and clad
// builds against both sides of that. Spelled here rather than in clad's
// compatibility header: that one includes clang for the plugin's sake, and a
// tool that links no clang cannot carry its inline definitions.
#if LLVM_VERSION_MAJOR < 20
using CladRecordKeeper = RecordKeeper;
#else
using CladRecordKeeper = const RecordKeeper;
#endif

enum class Action : std::uint8_t {
  GenAnalyses,
  GenAnalysisDescs,
  GenAnalysesDocs,
  GenDiagnostics,
  GenDiagnosticsDocs,
  GenOptions,
  GenOptionsDocs
};

// The parser writes it, so it is neither const nor hidden from the linker by
// anything but its own definition. Marked on the declarator rather than the
// line above, which clang-format is free to move away from it.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static cl::opt<Action> TheAction(
    cl::desc("What to render:"),
    cl::values(clEnumValN(Action::GenAnalyses, "gen-analyses",
                          "the analyses, as CLAD_ANALYSIS entries"),
               clEnumValN(Action::GenAnalysisDescs, "gen-analysis-descs",
                          "the constructs and the ways to miss them"),
               clEnumValN(Action::GenAnalysesDocs, "gen-analyses-docs",
                          "the page the reports send a reader to"),
               clEnumValN(Action::GenDiagnostics, "gen-diagnostics",
                          "the messages, as CLAD_DIAG entries"),
               clEnumValN(Action::GenDiagnosticsDocs, "gen-diagnostics-docs",
                          "the page listing what clad says"),
               clEnumValN(Action::GenOptions, "gen-options",
                          "the options, as CLAD_OPTION entries"),
               clEnumValN(Action::GenOptionsDocs, "gen-options-docs",
                          "the page listing the options")));

/// Says where the file came from and how to make it again. Every generated
/// file opens with it, in whichever comment its language spells: \p Open
/// starts the comment and \p Cont continues it.
static std::string banner(StringRef Open, StringRef Cont, StringRef Table,
                          StringRef Target) {
  return (Open + " Generated from " + Table + " by clad-tblgen.\n" + Cont +
          " Do not edit: edit the .td and regenerate with `ninja " + Target +
          "`.\n")
      .str();
}

/// The two tables, and the target that renders each.
static std::string cxxBanner(StringRef Table, StringRef Target) {
  return banner("//", "//", Table, Target);
}
static std::string rstBanner(StringRef Table, StringRef Target) {
  // An RST comment is continued by indenting, not by opening a second one.
  return banner("..", "  ", Table, Target);
}

/// The text as C++ string literals, wrapped so the generated line stays
/// readable. Every piece but the last keeps the space that joined it to the
/// next.
static std::string cxxString(StringRef Text) {
  std::vector<std::string> Lines;
  std::string Cur;
  while (!Text.empty()) {
    StringRef Word;
    std::tie(Word, Text) = Text.split(' ');
    if (!Cur.empty() && Cur.size() + 1 + Word.size() > 60) {
      Lines.push_back(Cur);
      Cur = Word.str();
    } else {
      if (Cur.empty())
        Cur = Word.str();
      else {
        Cur += ' ';
        Cur += Word;
      }
    }
  }
  Lines.push_back(Cur);

  std::string Out;
  for (unsigned I = 0, E = Lines.size(); I != E; ++I) {
    if (I)
      Out += "\n    ";
    Out += '"';
    for (char C : Lines[I]) {
      if (C == '"')
        Out += '\\';
      Out += C;
    }
    if (I + 1 != E)
      Out += ' ';
    Out += '"';
  }
  return Out;
}

/// The field's text, or nothing where the table left it unset. Spelled
/// through isValueUnset rather than getValueAsOptionalString, whose return
/// type changed from llvm::Optional to std::optional in LLVM 16, within the
/// range clad supports.
static StringRef optionalString(const Record* R, StringRef Field) {
  // Absent and unset are both "nothing here": DocBrief is clad's own, so only
  // the options that carry one have the field at all.
  const RecordVal* V = R->getValue(Field);
  if (!V || !V->getValue() || isa<UnsetInit>(V->getValue()))
    return StringRef();
  return R->getValueAsString(Field);
}

/// The options clad declares, by record name. OptParser.td contributes INPUT
/// and UNKNOWN to every table; neither is an option a user types.
static std::vector<const Record*> optionsOf(CladRecordKeeper& Records) {
  std::vector<const Record*> Out;
  for (const Record* R : Records.getAllDerivedDefinitions("Option")) {
    StringRef Kind = R->getValueAsDef("Kind")->getName();
    if (Kind != "KIND_INPUT" && Kind != "KIND_UNKNOWN")
      Out.push_back(R);
  }
  return Out;
}

/// How the option is spelled on a command line, prefix included.
static std::string spellingOf(const Record* R) {
  std::vector<StringRef> Prefixes = R->getValueAsListOfStrings("Prefixes");
  std::string Out = Prefixes.empty() ? std::string() : Prefixes.front().str();
  return Out + R->getValueAsString("Name").str();
}

/// The kind, spelled the way clad's macros name it.
static StringRef kindOf(const Record* R) {
  StringRef Kind = R->getValueAsDef("Kind")->getName();
  if (Kind == "KIND_FLAG")
    return "Flag";
  if (Kind == "KIND_JOINED")
    return "Joined";
  PrintFatalError(R->getLoc(), "clad takes no option of kind " + Kind);
}

/// What a marshalled option writes: the field's type, and the expression the
/// parser assigns. Read from the normalizer, which is what distinguishes the
/// marshalling classes OptParser.td ships.
struct Marshalling {
  StringRef Type;
  StringRef Value;
};

static Marshalling marshallingOf(const Record* R) {
  StringRef N = R->getValueAsString("Normalizer");
  if (N == "normalizeSimpleFlag")
    return {"bool", "true"};
  if (N == "normalizeSimpleNegativeFlag")
    return {"bool", "false"};
  if (N == "normalizeString")
    return {"std::string", "Val.str()"};
  if (N == "normalizeOptionalString")
    return {"std::optional<std::string>", "Val.str()"};
  PrintFatalError(R->getLoc(), "clad marshals no option through " + N);
}

/// Whether the option sets a field at all. -help and the deprecated
/// estimation-model switch do something instead.
static bool marshalled(const Record* R) {
  return !optionalString(R, "KeyPath").empty();
}

/// What a reader is told about the option: its DocBrief where it has one,
/// and otherwise the single line that goes on the help screen.
static StringRef proseOf(const Record* R) {
  StringRef Doc = optionalString(R, "DocBrief");
  return Doc.empty() ? R->getValueAsString("HelpText") : Doc;
}

/// The words of a text, whatever whitespace laid it out in the table. A
/// DocBrief is written as a code block, so it keeps the line breaks of the
/// .td, which say nothing about how the text should be rendered.
static SmallVector<StringRef, 64> wordsOf(StringRef Text) {
  SmallVector<StringRef, 64> Words;
  llvm::SplitString(Text, Words);
  return Words;
}

/// Text wrapped into lines of at most \p Width, each opened with \p Prefix.
static std::string wrapped(StringRef Text, StringRef Prefix, unsigned Width) {
  std::string Out;
  std::string Line = Prefix.str();
  for (StringRef Word : wordsOf(Text)) {
    // A word wider than the line still goes on one of its own: breaking it
    // would change what it says.
    if (Line.size() > Prefix.size() && Line.size() + 1 + Word.size() > Width) {
      Out += Line + "\n";
      Line = Prefix.str();
    }
    Line += (" " + Word).str();
  }
  return Out + Line + "\n";
}

/// The column limit less the two spaces an option's prose is indented by.
static constexpr unsigned RSTProseWidth = 76;

static void emitOptions(raw_ostream& OS, CladRecordKeeper& Records) {
  OS << cxxBanner("lib/Differentiator/Options.td", "clad-options") << R"(
#ifndef CLAD_OPTION
#error "define CLAD_OPTION(Id, Kind, Spelling, MetaVar, Help) before including"
#endif

// An option that also sets a field of clad::Options. Leave it undefined to
// see every option as a plain CLAD_OPTION. Value is the expression the field
// takes; for a Joined option it names Val, the text after the spelling.
#ifndef CLAD_OPTION_WITH_MARSHALLING
#define CLAD_OPTION_WITH_MARSHALLING(Id, Kind, Spelling, MetaVar, Help, Type,  \
                                     KeyPath, Default, Value)                  \
  CLAD_OPTION(Id, Kind, Spelling, MetaVar, Help)
#endif

)";
  for (const Record* R : optionsOf(Records)) {
    StringRef MetaVar = optionalString(R, "MetaVarName");
    std::string Head = ("(" + R->getName() + ", " + kindOf(R) + ", \"" +
                        spellingOf(R) + "\", \"" + MetaVar + "\"" + ",\n    ")
                           .str();
    if (!marshalled(R)) {
      OS << "CLAD_OPTION" << Head << cxxString(R->getValueAsString("HelpText"))
         << ")\n";
      continue;
    }
    Marshalling M = marshallingOf(R);
    OS << "CLAD_OPTION_WITH_MARSHALLING" << Head
       << cxxString(R->getValueAsString("HelpText")) << ",\n    " << M.Type
       << ", " << R->getValueAsString("KeyPath") << ", "
       << R->getValueAsString("DefaultValue") << ", " << M.Value << ")\n";
  }
  OS << "\n#undef CLAD_OPTION_WITH_MARSHALLING\n#undef CLAD_OPTION\n";
}

static void emitOptionsDocs(raw_ostream& OS, CladRecordKeeper& Records) {
  StringRef Title = "Options";
  OS << rstBanner("lib/Differentiator/Options.td", "clad-options") << "\n"
     << Title << "\n"
     << std::string(Title.size(), '*') << R"(

``-fplugin=`` both loads clad and runs it; each option below is then passed
to it behind its own ``-Xclang -plugin-arg-clad``::

  clang++ -fplugin=/path/to/clad.so \
          -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn ...

The driver spells the same thing without ``-Xclang``. Note the doubled
dash: clang appends whatever follows ``-fplugin-arg-clad-`` to the plugin's
arguments verbatim, and every option below starts with one of its own::

  clang++ -fplugin=/path/to/clad.so -fplugin-arg-clad--fdump-derived-fn ...

Either spelling works on every clang clad supports.

)";
  for (const Record* R : optionsOf(Records))
    // Spelt as the help screen spells it, metavariable included, so that the
    // two read the same.
    OS << "``" << spellingOf(R) << optionalString(R, "MetaVarName") << "``\n"
       << wrapped(proseOf(R), "  ", RSTProseWidth) << "\n";
}

/// What a list-of-def field names, in the order the table gives them.
static std::vector<const Record*> listOf(const Record* R, StringRef Field) {
  std::vector<const Record*> Out;
  for (const auto* D : R->getValueAsListOfDefs(Field))
    Out.push_back(D);
  return Out;
}

/// The reStructuredText of a code block field, without its trailing blank.
static std::string trimmed(StringRef Text) { return Text.rtrim().str(); }

/// Every record of class \p Class has to be reachable from Clad's \p Field.
/// One that is not is written down and never rendered -- no table entry, no
/// section in the documentation -- and the table alone cannot say so.
static void checkListed(CladRecordKeeper& Records, const Record* Clad,
                        StringRef Class, StringRef Field) {
  DenseSet<const Record*> Listed;
  for (const Record* R : listOf(Clad, Field))
    Listed.insert(R);
  for (const Record* R : Records.getAllDerivedDefinitions(Class))
    if (!Listed.contains(R))
      PrintFatalError(R->getLoc(), Twine("Clad's ") + Field +
                                       " does not list " + R->getName() +
                                       ", so nothing renders it");
}

/// What a code promises: a report prints it, a build log keeps it and a reader
/// searches for it, so two constructs must never share one. Every Desc is
/// looked at, listed or not, because a construct that has been retired still
/// owns its number.
static void checkCodes(CladRecordKeeper& Records) {
  DenseMap<int64_t, const Record*> ByCode;
  for (const Record* S : Records.getAllDerivedDefinitions("Desc")) {
    const Record*& Owner = ByCode[S->getValueAsInt("Code")];
    if (Owner)
      PrintFatalError(S->getLoc(), Twine("CLAD") +
                                       Twine(S->getValueAsInt("Code")) +
                                       " is already " + Owner->getName());
    Owner = S;
  }
}

/// Every way to miss a construct belongs to exactly one construct: a report
/// prints it under that construct's code, and the page lists it in that
/// construct's section.
static void checkMisses(CladRecordKeeper& Records, const Record* Clad) {
  DenseMap<const Record*, const Record*> Owner;
  for (const Record* S : listOf(Clad, "Descs"))
    for (const Record* M : listOf(S, "Misses")) {
      const Record*& First = Owner[M];
      if (First)
        PrintFatalError(M->getLoc(), Twine(M->getName()) + " is listed by " +
                                         First->getName() + " and " +
                                         S->getName());
      First = S;
    }
  for (const Record* M : Records.getAllDerivedDefinitions("Miss"))
    if (!Owner.lookup(M))
      PrintFatalError(M->getLoc(), Twine("no construct lists ") + M->getName());
}

/// What the table has to hold for the rest of this file to render all of it.
/// Checked before anything is written, so a mistake is reported once rather
/// than once per backend.
static void checkAnalyses(CladRecordKeeper& Records) {
  const Record* Clad = Records.getDef("Clad");
  if (!Clad)
    return;
  checkCodes(Records);
  checkListed(Records, Clad, "Analysis", "Analyses");
  checkListed(Records, Clad, "Diagnostic", "Diagnostics");
  checkListed(Records, Clad, "Desc", "Descs");
  checkMisses(Records, Clad);
}

static void emitAnalyses(raw_ostream& OS, CladRecordKeeper& Records) {
  OS << cxxBanner("lib/Differentiator/Analyses.td", "clad-analyses") << R"(
#ifndef CLAD_ANALYSIS
#error "define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc) before including"
#endif

)";
  for (const Record* A : listOf(Records.getDef("Clad"), "Analyses"))
    OS << "CLAD_ANALYSIS(" << A->getName() << ", \""
       << A->getValueAsString("Name") << "\", \""
       << A->getValueAsString("Legacy") << "\", "
       << (A->getValueAsBit("Default") ? "true" : "false") << ",\n    "
       << cxxString(A->getValueAsString("Summary")) << ")\n";
  OS << "\n#undef CLAD_ANALYSIS\n";
}

static void emitAnalysisDescs(raw_ostream& OS, CladRecordKeeper& Records) {
  OS << cxxBanner("lib/Differentiator/Analyses.td", "clad-analyses") << R"(
#ifndef CLAD_ANALYSIS_DESC
#error "define CLAD_ANALYSIS_DESC(Id, Analysis, Name, Code, Cost) before including"
#endif

#ifndef CLAD_ANALYSIS_MISS
#error "define CLAD_ANALYSIS_MISS(Id, Desc, Detail) before including"
#endif

)";
  for (const Record* S : listOf(Records.getDef("Clad"), "Descs")) {
    OS << "CLAD_ANALYSIS_DESC(" << S->getName() << ", "
       << S->getValueAsDef("Owner")->getName() << ", \""
       << S->getValueAsString("Name") << "\", " << S->getValueAsInt("Code")
       << ",\n    " << cxxString(S->getValueAsString("Cost")) << ")\n";
    for (const Record* M : listOf(S, "Misses"))
      OS << "CLAD_ANALYSIS_MISS(" << M->getName() << ", " << S->getName()
         << ",\n    " << cxxString(M->getValueAsString("Detail")) << ")\n";
    OS << "\n";
  }
  OS << "#undef CLAD_ANALYSIS_DESC\n#undef CLAD_ANALYSIS_MISS\n";
}

static void emitAnalysesDocs(raw_ostream& OS, CladRecordKeeper& Records) {
  const Record* Clad = Records.getDef("Clad");
  StringRef Title = Clad->getValueAsString("Title");
  OS << rstBanner("lib/Differentiator/Analyses.td", "clad-analyses") << "\n"
     << Title << "\n"
     << std::string(Title.size(), '*') << "\n"
     << trimmed(Clad->getValueAsString("Overview")) << "\n";

  for (const Record* S : listOf(Clad, "Descs")) {
    int64_t Code = S->getValueAsInt("Code");
    std::string Heading =
        ("CLAD" + Twine(Code) + ": " + S->getValueAsString("Name")).str();
    // The anchor is the code, which is what a report prints and a reader
    // searches for.
    OS << "\n.. _clad" << Code << ":\n\n"
       << Heading << "\n"
       << std::string(Heading.size(), '=') << "\n"
       << trimmed(S->getValueAsString("Doc")) << "\n";

    std::vector<const Record*> Misses = listOf(S, "Misses");
    if (Misses.empty())
      continue;
    OS << "\nWhat a report says when this is missed:\n\n";
    for (const Record* M : Misses)
      OS << "- " << M->getValueAsString("Detail") << "\n";
  }

  OS << R"(
Turning an analysis on or off

Each analysis is optional: it changes the code clad generates, never the
values that code computes. Turning one on asks clad to prove more and store
less; turning one off falls back to the conservative derivative.
``-fenable-analysis=<name>`` and ``-fdisable-analysis=<name>`` take the names
below, and each analysis also has a switch of its own.

)";
  for (const Record* A : listOf(Clad, "Analyses"))
    OS << "``-enable-" << A->getValueAsString("Legacy") << "`` / ``-disable-"
       << A->getValueAsString("Legacy") << "``\n"
       << "  Turns the " << A->getValueAsString("Name")
       << " analysis on or off for the whole translation unit, unless an\n"
       << "  individual request specifies otherwise. It "
       << A->getValueAsString("Summary")
       << ". Default: " << (A->getValueAsBit("Default") ? "on" : "off")
       << ".\n\n";
}

static void emitDiagnostics(raw_ostream& OS, CladRecordKeeper& Records) {
  OS << cxxBanner("lib/Differentiator/Analyses.td", "clad-analyses") << R"(
#ifndef CLAD_DIAG
#error "define CLAD_DIAG(Id, Severity, Text) before including"
#endif

)";
  for (const Record* D : listOf(Records.getDef("Clad"), "Diagnostics"))
    OS << "CLAD_DIAG(" << D->getName() << ", "
       << D->getValueAsString("Severity") << ",\n    "
       << cxxString(D->getValueAsString("Text")) << ")\n";
  OS << "\n#undef CLAD_DIAG\n";
}

static void emitDiagnosticsDocs(raw_ostream& OS, CladRecordKeeper& Records) {
  StringRef Title = "What clad says";
  OS << rstBanner("lib/Differentiator/Analyses.td", "clad-analyses") << "\n"
     << Title << "\n"
     << std::string(Title.size(), '*') << R"(

Every message clad emits, as it is worded. A placeholder like ``%0`` is
filled in where the message is emitted: a name from the code, or one of the
constructs in :doc:`Analyses`.

)";
  for (const Record* D : listOf(Records.getDef("Clad"), "Diagnostics")) {
    OS << "``" << D->getName() << "``\n"
       << "  " << D->getValueAsString("Severity") << ": "
       << D->getValueAsString("Text") << "\n\n";
  }
}

static bool cladTableGenMain(raw_ostream& OS, CladRecordKeeper& Records) {
  // Options.td has no Clad record, so this is a no-op for the option backends.
  checkAnalyses(Records);
  switch (TheAction.getValue()) {
  case Action::GenAnalyses:
    emitAnalyses(OS, Records);
    break;
  case Action::GenAnalysisDescs:
    emitAnalysisDescs(OS, Records);
    break;
  case Action::GenAnalysesDocs:
    emitAnalysesDocs(OS, Records);
    break;
  case Action::GenDiagnostics:
    emitDiagnostics(OS, Records);
    break;
  case Action::GenDiagnosticsDocs:
    emitDiagnosticsDocs(OS, Records);
    break;
  case Action::GenOptions:
    emitOptions(OS, Records);
    break;
  case Action::GenOptionsDocs:
    emitOptionsDocs(OS, Records);
    break;
  }
  return false;
}

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return TableGenMain(argv[0], &cladTableGenMain);
}
