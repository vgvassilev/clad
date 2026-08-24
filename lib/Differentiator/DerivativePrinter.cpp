#include "DerivativePrinter.h"

#include "clang/AST/Decl.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>

using namespace clang;

namespace clad {

namespace {

/// Notes where each statement's text begins as the printer reaches it.
///
/// PrinterHelper is a substitution hook: returning true means "I printed this,
/// skip it". Returning false leaves the output byte for byte what it would
/// have been, which is what makes the offsets describe the text a reader sees.
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

const DerivativePrinter::Rendering&
DerivativePrinter::render(const FunctionDecl* FD) {
  auto Found = m_Rendered.find(FD);
  if (Found != m_Rendered.end())
    return Found->second;

  Rendering R;
  // Not retained: the SourceManager keeps a copy for the life of the
  // compilation, and in a long-running session nothing is reclaimed until it
  // ends, so a second copy would double what this costs.
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

  SourceManager& SM = m_Sema.getSourceManager();
  std::string Name = "<clad derivative of " + FD->getNameAsString() + ">";
  // Entered from the differentiation that asked for it, falling back to the
  // main file. Somewhere inside the translation unit is required either way:
  // isBeforeInTranslationUnit orders two locations by walking up to a file
  // they share, and reports "Unsortable locations found" when there is none.
  SourceLocation IncludeLoc = m_RequestedAt.lookup(FD);
  if (IncludeLoc.isInvalid())
    IncludeLoc = SM.getLocForStartOfFile(SM.getMainFileID());
  R.File = SM.createFileID(llvm::MemoryBuffer::getMemBufferCopy(Text, Name),
                           SrcMgr::C_User, /*LoadedID=*/0, /*LoadedOffset=*/0,
                           IncludeLoc);
  return m_Rendered.insert({FD, std::move(R)}).first->second;
}

void DerivativePrinter::noteRequestedAt(const FunctionDecl* FD,
                                        SourceLocation Loc) {
  // The first request wins, and a later one is ignored rather than asserted
  // on: which request a shared derivative is attributed to is presentational,
  // and not worth ending a compilation over.
  if (Loc.isValid() && Loc.isFileID())
    m_RequestedAt.try_emplace(FD, Loc);
}

SourceLocation DerivativePrinter::locationOf(const FunctionDecl* FD,
                                             const Stmt* S) {
  const Rendering& R = render(FD);
  auto Found = R.Offsets.find(S);
  if (Found == R.Offsets.end())
    return {};
  // The printer announces a statement before emitting the indentation in
  // front of it, so the recorded offset can sit at the start of the line.
  // Point at the first character of the statement itself, which is where a
  // caret belongs.
  llvm::StringRef Text = m_Sema.getSourceManager().getBufferData(R.File);
  unsigned Offset = Found->second;
  while (Offset < Text.size() && (Text[Offset] == ' ' || Text[Offset] == '\t'))
    ++Offset;
  return m_Sema.getSourceManager()
      .getLocForStartOfFile(R.File)
      .getLocWithOffset(Offset);
}

SourceRange DerivativePrinter::rangeOf(const FunctionDecl* FD, const Stmt* S) {
  SourceLocation Begin = locationOf(FD, S);
  if (Begin.isInvalid())
    return {};
  // How wide the node is, measured by rendering it alone under the same
  // policy. A statement is printed with a trailing separator it does not own,
  // so that comes back off.
  std::string Own;
  llvm::raw_string_ostream OS(Own);
  S->printPretty(OS, /*Helper=*/nullptr,
                 m_Sema.getASTContext().getPrintingPolicy());
  OS.flush();
  llvm::StringRef Text = llvm::StringRef(Own).rtrim(" \t\n;");
  // Only a node that renders on one line can be measured this way: a compound
  // statement printed alone carries its own line breaks, and a range built
  // from that would end before it began. Those report a bare location, which
  // is right for them anyway -- underlining a whole loop body says nothing.
  if (Text.empty() || Text.contains(/*C=*/'\n'))
    return Begin;
  // A range's end names the last character, not one past it.
  return {Begin, Begin.getLocWithOffset(Text.size() - 1)};
}

llvm::StringRef DerivativePrinter::textOf(const FunctionDecl* FD) {
  return m_Sema.getSourceManager().getBufferData(render(FD).File);
}

} // namespace clad
