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

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
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

enum class Action : std::uint8_t { GenOptions, GenOptionsDocs };

// The parser writes it, so it is neither const nor hidden from the linker by
// anything but its own definition. Marked on the declarator rather than the
// line above, which clang-format is free to move away from it.
static cl::opt<Action>
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    TheAction(cl::desc("What to render:"),
              cl::values(clEnumValN(Action::GenOptions, "gen-options",
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

From clang 14 the driver spells it without ``-Xclang``. Note the doubled
dash: clang appends whatever follows ``-fplugin-arg-clad-`` to the plugin's
arguments verbatim, and every option below starts with one of its own::

  clang++ -fplugin=/path/to/clad.so -fplugin-arg-clad--fdump-derived-fn ...

Clad supports clang 12 and later, and clang 12 and 13 have no
``-fplugin-arg-``; the ``-Xclang`` form above works on every version.

)";
  for (const Record* R : optionsOf(Records))
    // Spelt as the help screen spells it, metavariable included, so that the
    // two read the same.
    OS << "``" << spellingOf(R) << optionalString(R, "MetaVarName") << "``\n"
       << wrapped(proseOf(R), "  ", RSTProseWidth) << "\n";
}

static bool cladTableGenMain(raw_ostream& OS, CladRecordKeeper& Records) {
  switch (TheAction.getValue()) {
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
