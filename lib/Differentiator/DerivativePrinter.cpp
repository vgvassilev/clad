#include "DerivativePrinter.h"

#include "clang/AST/Decl.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

using namespace clang;

namespace clad {

namespace {

/// Notes where each statement's text begins as the printer reaches it.
///
/// PrinterHelper is a substitution hook: returning true means "I printed
/// this, skip it". Returning false leaves the output exactly as it would have
/// been, so the offsets describe the text a reader sees.
class OffsetRecorder : public PrinterHelper {
  llvm::DenseMap<const Stmt*, unsigned>& m_Offsets;

public:
  explicit OffsetRecorder(llvm::DenseMap<const Stmt*, unsigned>& Offsets)
      : m_Offsets(Offsets) {}

  bool handledStmt(Stmt* S, llvm::raw_ostream& OS) override {
    // A node reached twice keeps the first offset, where its text starts.
    m_Offsets.try_emplace(S, static_cast<unsigned>(OS.tell()));
    return false;
  }
};

} // namespace

const DerivativePrinter::Printout&
DerivativePrinter::print(const FunctionDecl* FD) {
  auto Found = m_Printed.find(FD);
  if (Found != m_Printed.end())
    return Found->second;

  Printout R;
  std::string Text;
  llvm::raw_string_ostream OS(Text);
  PrintingPolicy Policy = m_Sema.getASTContext().getPrintingPolicy();

  // The signature comes from Decl::print, which takes no PrinterHelper, so it
  // is printed without one and the body follows separately. TerseOutput is
  // what stops Decl::print from printing the body itself.
  PrintingPolicy Signature = Policy;
  Signature.TerseOutput = true;
  FD->print(OS, Signature);
  OS << " ";

  if (const Stmt* Body = FD->getBody()) {
    OffsetRecorder Recorder(R.Offsets);
    Body->printPretty(OS, &Recorder, Policy);
  }
  OS.flush();

  // Into the chunk this function's own slots came from: a note on a slot can
  // only name a line of its own file. The body's opening brace is among the
  // first slots the function was given, so it names the chunk the function
  // started in. A derivative with no slot of its own takes a fresh one, which
  // at least guarantees there is a chunk to print into.
  SourceLocation Anchor;
  if (const Stmt* Body = FD->getBody())
    Anchor = Body->getBeginLoc();
  if (!m_Locs.owns(Anchor))
    Anchor = FD->getEndLoc();
  if (!m_Locs.owns(Anchor))
    Anchor = m_Locs.nextLoc();

  R.Size = static_cast<unsigned>(Text.size());
  R.At = m_Locs.addText(Anchor, Text);
  if (R.At) {
    R.LineStarts.push_back(0);
    for (unsigned I = 0; I != R.Size; ++I)
      if (Text[I] == '\n')
        R.LineStarts.push_back(I + 1);
  }
  return m_Printed.insert({FD, std::move(R)}).first->second;
}

/// Where \p S starts in \p Text, past the indentation the printer put in
/// front of it, or nothing if \p Offsets does not have \p S. A free function
/// so that the header need not name std::optional: whether that is reachable
/// there depends on the standard library, and on some it is not.
static std::optional<unsigned>
textOffsetOf(const llvm::DenseMap<const Stmt*, unsigned>& Offsets,
             llvm::StringRef Text, const Stmt* S) {
  auto Found = Offsets.find(S);
  if (Found == Offsets.end())
    return std::nullopt;
  // The printer announces a statement before the indentation in front of it,
  // so the recorded offset can sit at the start of the line. Both callers
  // want the statement itself: a caret belongs on its first character, and so
  // does the text reported beside a position.
  unsigned Offset = Found->second;
  while (Offset < Text.size() && (Text[Offset] == ' ' || Text[Offset] == '\t'))
    ++Offset;
  return Offset;
}

SourceLocation DerivativePrinter::locationOf(const FunctionDecl* FD,
                                             const Stmt* S) {
  const Printout& R = print(FD);
  if (!R.At)
    return {};
  std::optional<unsigned> Offset =
      textOffsetOf(R.Offsets, m_Locs.textAt(R.At, R.Size), S);
  if (!Offset)
    return {};
  return R.At.Loc.getLocWithOffset(static_cast<int>(*Offset));
}

unsigned DerivativePrinter::lineOf(const FunctionDecl* FD, const Stmt* S) {
  const Printout& R = print(FD);
  if (!R.At)
    return 0;
  auto Found = R.Offsets.find(S);
  if (Found == R.Offsets.end())
    return 0;
  // Counted from the line the printout starts at. Asking the SourceManager
  // instead is not an option: it settles a chunk's lines on the first
  // question, and more code may still be printed into the chunk after that.
  const auto It =
      std::upper_bound(R.LineStarts.begin(), R.LineStarts.end(), Found->second);
  return R.At.Line + static_cast<unsigned>(It - R.LineStarts.begin()) - 1;
}

unsigned DerivativePrinter::lineOf(const FunctionDecl* FD) {
  return print(FD).At.Line;
}

SourceRange DerivativePrinter::rangeOf(const FunctionDecl* FD, const Stmt* S) {
  SourceLocation Begin = locationOf(FD, S);
  if (Begin.isInvalid())
    return {};
  // How wide the node is, measured by printing it alone under the same
  // policy. A statement carries a trailing separator it does not own, so that
  // comes back off.
  std::string Own;
  llvm::raw_string_ostream OS(Own);
  S->printPretty(OS, /*Helper=*/nullptr,
                 m_Sema.getASTContext().getPrintingPolicy());
  OS.flush();
  llvm::StringRef Text = llvm::StringRef(Own).rtrim(" \t\n;");
  // Only a node that prints on one line can be measured this way: one with
  // line breaks would give a range that ends before it begins. Those report a
  // bare location, which is right anyway -- underlining a loop body says
  // nothing.
  if (Text.empty() || Text.contains(/*C=*/'\n'))
    return Begin;
  // A range's end names the last character, not one past it.
  return {Begin, Begin.getLocWithOffset(static_cast<int>(Text.size()) - 1)};
}

unsigned DerivativePrinter::offsetOf(const FunctionDecl* FD, const Stmt* S) {
  const Printout& R = print(FD);
  return textOffsetOf(R.Offsets, m_Locs.textAt(R.At, R.Size), S).value_or(0);
}

SourceLocation DerivativePrinter::startOf(const FunctionDecl* FD) {
  return print(FD).At.Loc;
}

llvm::StringRef DerivativePrinter::textOf(const FunctionDecl* FD) {
  const Printout& R = print(FD);
  return m_Locs.textAt(R.At, R.Size);
}

} // namespace clad
