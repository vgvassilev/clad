#include "GeneratedCode.h"

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Sema.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <system_error>
#include <utility>

using namespace clang;

namespace clad {

GeneratedCode::Chunk& GeneratedCode::makeChunk(unsigned TextBytes) {
  SourceManager& SM = m_Sema.getSourceManager();
  const std::string Name = nextName();
  auto Owned = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(
      TextBytes + kSlotsPerChunk, Name);
  char* Start = Owned->getBufferStart();
  // A blank line everywhere until code is printed over it: every slot has to
  // be a line of its own, and uninitialised bytes would otherwise reach
  // whoever reads this.
  std::memset(Start, '\n', TextBytes + kSlotsPerChunk);

  // Entered from the main file, the way clang-repl enters its input. A chunk
  // outside the include tree cannot be ordered against anything inside it:
  // isBeforeInTranslationUnit runs out of shared parents and reports
  // "Unsortable locations found", which clad hits whenever it sorts a
  // derivative against a pragma.
  Chunk C;
  C.File = SM.createFileID(std::move(Owned), SrcMgr::C_User,
                           /*LoadedID=*/0, /*LoadedOffset=*/0,
                           SM.getLocForStartOfFile(SM.getMainFileID()));
  C.Name = Name;
  C.Text = Start;
  C.Capacity = TextBytes;
  m_ChunkList.push_back(std::move(C));
  return m_ChunkList.back();
}

SourceLocation GeneratedCode::nextLoc() {
  if (m_Used == kSlotsPerChunk) {
    makeChunk(kSlotsPerChunk * kTextBytesPerSlot);
    m_Used = 0;
  }
  Chunk& C = m_ChunkList.back();
  ++C.Slots;
  // Slots sit behind the code. Signed because that is what getLocWithOffset
  // takes; a chunk is small enough that the offset always fits.
  return m_Sema.getSourceManager()
      .getLocForStartOfFile(C.File)
      .getLocWithOffset(static_cast<int>(C.Capacity + m_Used++));
}

void GeneratedCode::setFileBase(llvm::StringRef Dir, llvm::StringRef Unit) {
  if (Unit.empty())
    return; // Nothing to name them after; leave them anonymous.
  // Left exactly as it was given. A relative name is written relative to the
  // working directory and read back against the compilation directory the
  // line table records, which is the same place; making it absolute here
  // would only put this machine's paths in the debug information.
  llvm::SmallString<128> Path(Dir);
  llvm::sys::path::append(Path, llvm::sys::path::filename(Unit));
  m_FileBase = std::string(Path);
}

std::string GeneratedCode::nextName() const {
  // Named apart from the chunks before it. Every chunk numbers its lines
  // from 1 and the line table keys a file by name, so two chunks sharing a
  // name would put two statements at the same position.
  const size_t N = m_ChunkList.size() + 1;
  if (m_FileBase.empty())
    return N == 1 ? "<clad generated code>"
                  : "<clad generated code #" + std::to_string(N) + ">";
  return N == 1 ? m_FileBase + ".clad.cpp"
                : m_FileBase + ".clad." + std::to_string(N) + ".cpp";
}

void GeneratedCode::writeChunksToFiles(DiagnosticsEngine& Diags) const {
  if (m_FileBase.empty())
    return;
  for (const Chunk& C : m_ChunkList) {
    if (!C.Used)
      continue;
    std::error_code EC;
    llvm::raw_fd_ostream Out(C.Name, EC, llvm::sys::fs::OF_Text);
    if (EC) {
      unsigned ID = Diags.getCustomDiagID(
          DiagnosticsEngine::Warning,
          "could not write the generated source to '%0': %1");
      Diags.Report(ID) << C.Name << EC.message();
      continue;
    }
    Out << llvm::StringRef(C.Text, C.Used);
  }
}

GeneratedCode::Chunk* GeneratedCode::chunkFor(SourceLocation Loc) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  return const_cast<Chunk*>(
      static_cast<const GeneratedCode*>(this)->chunkFor(Loc));
}

const GeneratedCode::Chunk* GeneratedCode::chunkFor(SourceLocation Loc) const {
  if (Loc.isInvalid())
    return nullptr;
  FileID File = m_Sema.getSourceManager().getFileID(Loc);
  for (const Chunk& C : m_ChunkList)
    if (C.File == File)
      return &C;
  return nullptr;
}

GeneratedCode::Placement GeneratedCode::addText(SourceLocation Anchor,
                                                llvm::StringRef Text) {
  Chunk* C = chunkFor(Anchor);
  if (!C || Text.empty())
    return {};
  SourceManager& SM = m_Sema.getSourceManager();
  // A trailing newline so the next derivative starts on a line of its own
  // rather than continuing this one.
  const size_t Need = Text.size() + 1;
  if (Need > C->Capacity - C->Used) {
    // A derivative with more nodes than a chunk holds runs its slots on into
    // the next chunks, and its code can outgrow the room in any one of them.
    // Give it a chunk of its own, big enough. Its slots stay where they were,
    // so a note cannot name a line of this one and only the diagnostics get
    // the code; that is still better than a derivative nothing can show.
    C = &makeChunk(static_cast<unsigned>(Need));
    // The next slot starts a chunk again, so nothing is handed out of this
    // one and it stays what it is: somewhere to put code.
    m_Used = kSlotsPerChunk;
  }
  const unsigned At = C->Used;
  const unsigned First = C->NextLine;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  std::memcpy(C->Text + At, Text.data(), Text.size());
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  C->Text[At + Text.size()] = '\n';
  C->Used += static_cast<unsigned>(Need);
  // Printing over blank lines removes newlines, so the lines after this one
  // move up. Only lines nothing has been printed on yet, so nothing already
  // placed moves.
  C->NextLine += static_cast<unsigned>(Text.count('\n')) + 1;
  return {
      SM.getLocForStartOfFile(C->File).getLocWithOffset(static_cast<int>(At)),
      First};
}

llvm::StringRef GeneratedCode::textAt(Placement P, unsigned Size) const {
  const Chunk* C = chunkFor(P.Loc);
  if (!C)
    return {};
  const unsigned At = m_Sema.getSourceManager().getFileOffset(P.Loc);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  return llvm::StringRef(C->Text + At, std::min(Size, C->Capacity - At));
}

void GeneratedCode::assign(SourceLocation Begin, SourceLocation End,
                           unsigned Line) {
  Chunk* C = chunkFor(Begin);
  if (!C || !Line)
    return;
  SourceManager& SM = m_Sema.getSourceManager();
  const unsigned First = SM.getFileOffset(Begin);
  // An end in another chunk says nothing about this one; keep to the slot
  // the range started at.
  unsigned Last = First;
  if (End.isValid() && chunkFor(End) == C)
    Last = std::max(First, SM.getFileOffset(End));

  // Slots only: the range comes from nodes, which never sit in the code.
  C->SlotLine.resize(C->Slots, 0);
  for (unsigned O = First; O <= Last; ++O) {
    const unsigned I = O - C->Capacity;
    if (I < C->SlotLine.size())
      C->SlotLine[I] = Line;
  }
}

void GeneratedCode::forgetLineTables() {
  SourceManager& SM = m_Sema.getSourceManager();
  // The line table is a cache the SourceManager fills on demand, so dropping
  // it only means the next question is answered by reading the buffer again.
  for (const Chunk& C : m_ChunkList)
    SM.getSLocEntry(C.File).getFile().getContentCache().SourceLineCache =
        SrcMgr::LineOffsetMapping();
  // It also remembers the answer it last gave, to start the next search near
  // it. That answer came from a table just dropped, so ask about another file
  // to make the next question about these start over.
  SM.getLineNumber(SM.getMainFileID(), 0);
}

void GeneratedCode::present() {
  SourceManager& SM = m_Sema.getSourceManager();
  for (Chunk& C : m_ChunkList) {
    if (C.Presented)
      continue; // Its notes are in, and a second pass would go backwards.
    C.Presented = true;
    C.SlotLine.resize(C.Slots, 0);
    const SourceLocation Base = SM.getLocForStartOfFile(C.File);
    // Line 1 until told otherwise, so a slot nobody described points at the
    // start of the code rather than past its end.
    unsigned Line = 1;
    for (unsigned I = 0; I != C.Slots; ++I) {
      if (C.SlotLine[I])
        Line = C.SlotLine[I];
      // Line + 1 because a note reads as #line does: it gives the number of
      // the line after it, and each slot is a line.
      SM.AddLineNote(Base.getLocWithOffset(static_cast<int>(C.Capacity + I)),
                     Line + 1,
                     /*FilenameID=*/-1, /*IsFileEntry=*/false,
                     /*IsFileExit=*/false, SrcMgr::C_User);
    }
  }
}

} // namespace clad
