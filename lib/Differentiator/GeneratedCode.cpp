#include "GeneratedCode.h"

#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Sema.h"

#include "llvm/Support/MemoryBuffer.h"

#include <string>

using namespace clang;

namespace clad {

SourceLocation GeneratedCode::nextLoc() {
  SourceManager& SM = m_Sema.getSourceManager();
  if (m_Used == kSlotsPerChunk) {
    // One newline per slot: a slot is a line, so its presumed line is already
    // its index and a note only has to say where that line really is.
    std::string Slots(kSlotsPerChunk, '\n');
    // Named apart from the chunks before it. A chunk restarts line numbering
    // at 1 and the line table keys a file by its name, so same-named chunks
    // fold into one entry and two generated statements report the same
    // position.
    std::string Name = "<clad generated code>";
    if (m_Chunks)
      Name = "<clad generated code #" + std::to_string(m_Chunks + 1) + ">";
    ++m_Chunks;
    // Entered from the main file. A chunk outside the translation unit's
    // include tree cannot be ordered against anything inside it:
    // isBeforeInTranslationUnit walks up to a file the two share, runs out of
    // parents, and reports "Unsortable locations found" -- which clad reaches
    // whenever it sorts a derivative against a pragma. clang-repl enters its
    // input buffers the same way.
    m_Chunk = SM.createFileID(llvm::MemoryBuffer::getMemBufferCopy(Slots, Name),
                              SrcMgr::C_User,
                              /*LoadedID=*/0, /*LoadedOffset=*/0,
                              SM.getLocForStartOfFile(SM.getMainFileID()));
    m_Used = 0;
  }
  return SM.getLocForStartOfFile(m_Chunk).getLocWithOffset(m_Used++);
}

} // namespace clad
