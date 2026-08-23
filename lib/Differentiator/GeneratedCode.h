#ifndef CLAD_DIFFERENTIATOR_GENERATEDCODE_H
#define CLAD_DIFFERENTIATOR_GENERATEDCODE_H

#include "clang/Basic/SourceLocation.h"

namespace clang {
class Sema;
} // namespace clang

namespace clad {

/// The code clad generates, and the source locations that point into it.
///
/// A location has to be supplied when a node is built, long before the text
/// that node will be printed as exists, so it cannot be the node's real
/// position. What it can be is *distinct*, which is enough to tell two
/// generated nodes apart in a diagnostic or in DWARF, and enough for a line
/// note to say later where each one really ended up.
///
/// Slots come from buffers of newlines, one per line, so a slot's presumed
/// line is its index. They are allocated in chunks as slots run out. One large
/// buffer instead would not do: a buffer handed to the SourceManager is fixed,
/// and reserving generously is not free either, because every slot is written
/// and the SourceManager scans a buffer end to end to build its line table the
/// first time anyone asks for a line number in it.
class GeneratedCode {
  /// Slots per buffer. A chunk is paid for in full once allocated, so this
  /// trades memory against how many files a derivative is split across -- the
  /// first is not allocated until the first slot is asked for.
  static constexpr unsigned kSlotsPerChunk = 8192;

  clang::Sema& m_Sema;
  clang::FileID m_Chunk;
  unsigned m_Used = kSlotsPerChunk; // forces the first nextLoc() to allocate
  unsigned m_Chunks = 0; // names the next chunk apart from the ones before it

public:
  explicit GeneratedCode(clang::Sema& S) : m_Sema(S) {}

  /// A location no other generated node has been given.
  clang::SourceLocation nextLoc();
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_GENERATEDCODE_H
