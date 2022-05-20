/// \file
/// This file contains data structures and typedef declarations for types
/// that are useful for parsing and storing differentiation arguments.

#ifndef CLAD_PARSE_DIFF_ARGS_TYPES_H
#define CLAD_PARSE_DIFF_ARGS_TYPES_H

#include "clang/AST/Decl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

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
  
  using IndexIntervalTable = llvm::SmallVector<IndexInterval, 16>;

  /// `DiffInputVarInfo` is designed to store all the essential information about a
  /// differentiation input variable. Please note that here input variable corresponds
  /// to mathematical variable, not a programming one. 
  // FIXME: 'DiffInputVarInfo' name is probably not accurate, since we can have multiple
  // differentiation input variables for same parameter as well. 'DiffInputVarInfo' 
  // name implicitly guides that there would be at most one `DiffInputVarInfo` object for
  // one parameter, but that is not strictly true.
  struct DiffInputVarInfo {
    /// Source string specified by user that defines differentiation
    /// specification for the input variable.
    /// For example, if complete input string specified by user is:
    /// 'u, v.first, arr[3]'
    /// then `source` data member value for 2nd input variable should be
    /// 'v.first'
    std::string source;
    /// Parameter associated with the input variable.
    const clang::ValueDecl* param = nullptr;
    /// array index range associated with the parameter.
    IndexInterval paramIndexInterval;
    /// Nested field information.
    llvm::SmallVector<std::string, 4> fields;
    // FIXME: Add support for differentiating with respect to array fields.
    // llvm::SmallVector<IndexInterval> fieldIndexIntervals;

    DiffInputVarInfo(const clang::ValueDecl* pParam = nullptr,
                     IndexInterval pParamIndexInterval = {},
                     llvm::SmallVector<std::string, 4> pFields = {})
        : param(pParam), paramIndexInterval(pParamIndexInterval),
          fields(pFields) {}

    // FIXME: Move function definitions to ParseDiffArgTypes.cpp
    bool operator==(const DiffInputVarInfo& rhs) const {
      return param == rhs.param &&
             paramIndexInterval == rhs.paramIndexInterval &&
             fields == rhs.fields;
    }
  };

  using DiffInputVarsInfo = llvm::SmallVector<DiffInputVarInfo, 16>;

  using DiffParams = llvm::SmallVector<const clang::ValueDecl*, 16>;
  using DiffParamsWithIndices = std::pair<DiffParams, IndexIntervalTable>;
  } // namespace clad

#endif