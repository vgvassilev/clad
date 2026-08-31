#ifndef CLAD_DIFFERENTIATOR_GENERATEDCODE_H
#define CLAD_DIFFERENTIATOR_GENERATEDCODE_H

#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <vector>

namespace clang {
class Sema;
} // namespace clang

namespace clad {

/// Somewhere for clad's generated code to be, and for its nodes to point at.
///
/// Nothing clad builds comes from a file, so a diagnostic about it has
/// nothing to underline and a debugger nothing to show. So clad prints each
/// derivative into a buffer the SourceManager knows about, and the nodes it
/// built point into that same buffer.
///
/// A node needs its location the moment it is built, long before the
/// derivative is printed, so it cannot be told where it will end up. It gets
/// a *slot* instead: one byte of the buffer, reserved for it alone. Once the
/// code is printed, a line note in front of each slot says which line that
/// node landed on, the way `#line` does.
///
/// The buffer -- a *chunk* -- is the printed code followed by the slots:
///
/// \code
///   inline void f_grad(double x, double *_d_x) {   line 1
///       double _t0 = t;                            line 2
///       t = t * t;                                 line 3
///   }                                              line 4
///                                                  a slot, presented as the
///                                                  line its node was printed
///                                                  on
/// \endcode
///
/// Code and slots share a file because a diagnostic and a debugger have to
/// agree on where a statement is. Slots come after the code, so one that
/// nobody described lands past the end of it rather than in the middle, and
/// take a byte each, since a slot needs a distinct offset and not a line.
///
/// Chunks are a fixed size, and a new one is made when the slots run out: the
/// SourceManager owns a buffer once it is handed over, so none of them can
/// grow.
///
/// The SourceManager also works out where a buffer's lines are on the first
/// question about it and keeps the answer, so code printed after that would
/// sit on lines it has never heard of. Clad cannot stop the question being
/// asked -- building a derivative puts generated locations in front of Sema,
/// and on some builds that is enough for something to ask. So the answer is
/// dropped instead: print everything, call forgetLineTables, then read.
class GeneratedCode {
  /// Slots in a chunk. A chunk is paid for in full as soon as it is made, so
  /// this trades memory against how many buffers a derivative is spread over.
  static constexpr unsigned kSlotsPerChunk = 8192;

  /// Room for printed code, per slot. Reserved up front because a buffer
  /// cannot grow, and paid for whether it is used or not. Clad's own test
  /// suite averages ten bytes of code per slot; a derivative that runs over
  /// what is left of a chunk does not get printed at all.
  static constexpr unsigned kTextBytesPerSlot = 12;

  /// One buffer: the printed code at the front, then one byte per slot.
  struct Chunk {
    clang::FileID File;
    std::string Name;       ///< what the line table calls it
    char* Text = nullptr;   ///< the writable region, at the front
    unsigned Capacity = 0;  ///< how many bytes of it there are
    unsigned Used = 0;      ///< how many of them are written
    unsigned Slots = 0;     ///< how many slots have been handed out
    unsigned NextLine = 1;  ///< the line the next code written here starts on
    bool Presented = false; ///< whether its line notes are already in
    /// The line of printed code each slot stands for, by slot; zero for a slot
    /// nobody described.
    std::vector<unsigned> SlotLine;
  };

  /// The chunk \p Loc is in, or null if it is not one of ours.
  [[nodiscard]] Chunk* chunkFor(clang::SourceLocation Loc);
  [[nodiscard]] const Chunk* chunkFor(clang::SourceLocation Loc) const;

  /// What to call the chunk about to be made.
  [[nodiscard]] std::string nextName() const;

  clang::Sema& m_Sema;
  llvm::SmallVector<Chunk, 2> m_ChunkList;
  unsigned m_Used = kSlotsPerChunk; // forces the first nextLoc() to allocate

public:
  explicit GeneratedCode(clang::Sema& S) : m_Sema(S) {}

  /// A location no other generated node has been given.
  clang::SourceLocation nextLoc();

  /// Where a piece of code was printed, and the line it starts on.
  struct Placement {
    clang::SourceLocation Loc; ///< the first character of the code
    unsigned Line = 0;         ///< the line it begins on; 0 if it went nowhere
    explicit operator bool() const { return Line != 0; }
  };

  /// Prints \p Text into the chunk \p Anchor is in, after whatever is there.
  ///
  /// \p Anchor is a slot of the code being printed: a note on a slot can only
  /// name a line of its own file, so the two have to share a chunk. Comes
  /// back empty if the code does not fit.
  Placement addText(clang::SourceLocation Anchor, llvm::StringRef Text);

  /// The bytes printed at \p P, \p Size of them.
  [[nodiscard]] llvm::StringRef textAt(Placement P, unsigned Size) const;

  /// Says that the slots from \p Begin through \p End belong to code on line
  /// \p Line of the chunk they are in.
  ///
  /// By range, because a statement is built out of many nodes and the one an
  /// instruction is attributed to is rarely the first. Assign the outer
  /// statements before the inner ones, so the innermost -- the most specific
  /// answer -- is the one that stays.
  void assign(clang::SourceLocation Begin, clang::SourceLocation End,
              unsigned Line);

  /// Puts a line note in front of every slot, so each presents as the line of
  /// code it stands for.
  ///
  /// Every slot, not only the described ones: without a note of its own a
  /// slot reports one line further on than the slot before it, which walks
  /// off the end of the code.
  void present();

  /// Throws away what the SourceManager worked out about these buffers, so
  /// that the next question about a line is answered from what they hold now.
  /// Call it once everything is printed and before anything reads.
  void forgetLineTables();

  /// Stops anything more being printed into the chunks made so far, because
  /// they have been read. Incremental compilation comes back for another
  /// round of derivatives; those get chunks of their own.
  void seal() { m_Used = kSlotsPerChunk; }

  /// Whether \p A and \p B are in the same chunk, which is what it takes for
  /// a note on one to name a line of the other.
  [[nodiscard]] bool isSameChunk(clang::SourceLocation A,
                                 clang::SourceLocation B) const {
    const Chunk* C = chunkFor(A);
    return C && C == chunkFor(B);
  }

  /// Whether \p Loc is one of the slots handed out here.
  [[nodiscard]] bool owns(clang::SourceLocation Loc) const {
    return chunkFor(Loc) != nullptr;
  }
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_GENERATEDCODE_H
