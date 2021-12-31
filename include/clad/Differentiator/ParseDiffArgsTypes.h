/// \file
/// This file contains data structures and typedef declarations for types
/// that are useful for parsing and storing differentiation arguments.

#ifndef CLAD_PARSE_DIFF_ARGS_TYPES_H
#define CLAD_PARSE_DIFF_ARGS_TYPES_H

#include "clang/AST/Decl.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <utility>


namespace clad {

  /// This class representa a range of integers. It is currently used to
  /// represent indices range of independent array parameters.
  struct IndexInterval {
    std::size_t Start;
    std::size_t Finish;

    IndexInterval() : Start(0), Finish(0) {}

    IndexInterval(std::size_t first, std::size_t last) : Start(first), Finish(last + 1) {}

    IndexInterval(std::size_t index) : Start(index), Finish(index + 1) {}

    std::size_t size() { return Finish - Start; }

    bool isInInterval(std::size_t n) { return n >= Start && n <= Finish; }

    bool operator==(const IndexInterval& rhs) const {
      return Start == rhs.Start && Finish == rhs.Finish;
    }
  };

  using DiffParams = llvm::SmallVector<const clang::ValueDecl*, 16>;
  using IndexIntervalTable = llvm::SmallVector<IndexInterval, 16>;
  using DiffParamsWithIndices = std::pair<DiffParams, IndexIntervalTable>;
} // namespace clad

#endif