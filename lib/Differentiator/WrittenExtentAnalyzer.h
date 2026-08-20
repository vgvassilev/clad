#ifndef CLAD_DIFFERENTIATOR_WRITTENEXTENTANALYZER_H
#define CLAD_DIFFERENTIATOR_WRITTENEXTENTANALYZER_H

#include "clang/AST/Decl.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace clad {

/// The range of one pointer parameter that a function writes, described in
/// terms of that function's own parameters so a call site can evaluate it by
/// substituting arguments.
///
/// The description over-approximates: it names a range that contains every
/// write, never one that misses any. A caller may therefore record more than
/// the callee strictly touches, but never less.
struct WrittenExtent {
  enum class Kind : std::uint8_t {
    /// The function does not write through this parameter. Nothing to record.
    None,
    /// Exactly one element, at a constant offset.
    Element,
    /// Elements [0, Bound), where Bound is a parameter or a constant.
    Range,
    /// The function writes through this parameter but the range could not be
    /// bounded. Callers must assume nothing and keep the conservative
    /// protocol.
    Unknown
  };

  Kind K = Kind::None;
  /// For Kind::Element: the constant offset written.
  std::uint64_t Offset = 0;
  /// For Kind::Range: whether Bound names a parameter or is a constant.
  bool BoundIsParam = false;
  /// For Kind::Range with BoundIsParam: index of the bounding parameter.
  unsigned BoundParamIdx = 0;
  /// For Kind::Range without BoundIsParam: the constant bound.
  std::uint64_t BoundConst = 0;

  [[nodiscard]] bool isProven() const { return K != Kind::Unknown; }
};

/// Computes, for each parameter of `FD`, the extent of that parameter the
/// function writes. The result is indexed by parameter position.
///
/// This is deliberately a small whitelist of shapes that can be proven by
/// inspection -- a constant subscript, a dereference, and a subscript by the
/// induction variable of an enclosing counted loop. Everything else, including
/// every write clad cannot attribute to a parameter, yields Kind::Unknown, so
/// a caller that gates on isProven() stays conservative as the whitelist
/// grows.
llvm::SmallVector<WrittenExtent, 8>
computeWrittenExtents(const clang::FunctionDecl* FD);

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_WRITTENEXTENTANALYZER_H
