//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang Automatic Differentiator
//----------------------------------------------------------------------------//

#ifndef CLAD_DIFFERENTIATOR_ANALYSES_H
#define CLAD_DIFFERENTIATOR_ANALYSES_H

#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

#include <cassert>
#include <cstdint>
#include <utility>

namespace clad {

/// The analyses clad can run, one per entry in Analyses.td.
enum class AnalysisId : std::uint8_t {
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, FirstBit, Desc) Id,
#include "clad/Differentiator/Analyses.def"
};

/// A construct an analysis can act on, one per CLAD_ANALYSIS_DESC in
/// Analyses.td.
enum class AnalysisDesc : std::uint8_t {
#define CLAD_ANALYSIS_DESC(Id, Analysis, Name, Code, Cost) Id,
#define CLAD_ANALYSIS_MISS(Id, Desc, Detail)
#include "AnalysisDescs.def"
};

/// A way an input can miss a construct, one per CLAD_ANALYSIS_MISS in
/// Analyses.td.
///
/// None means no miss was recorded. It is there for a record that carries a
/// miss beside an answer; a Proven never holds it.
enum class AnalysisMiss : std::uint8_t {
  None,
#define CLAD_ANALYSIS_DESC(Id, Analysis, Name, Code, Cost)
#define CLAD_ANALYSIS_MISS(Id, Desc, Detail) Id,
#include "AnalysisDescs.def"
};

/// How the analysis is named to users, as -fdisable-analysis takes it.
inline const char* nameOf(AnalysisId A) {
  switch (A) {
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, FirstBit, Desc)               \
  case AnalysisId::Id:                                                         \
    return Name;
#include "clad/Differentiator/Analyses.def"
  }
  llvm_unreachable("unhandled analysis"); // LCOV_EXCL_LINE
}

/// How the construct is named to users.
inline const char* nameOf(AnalysisDesc S) {
  switch (S) {
#define CLAD_ANALYSIS_DESC(Id, Analysis, Name, Code, Cost)                     \
  case AnalysisDesc::Id:                                                       \
    return Name;
#define CLAD_ANALYSIS_MISS(Id, Desc, Detail)
#include "AnalysisDescs.def"
  }
  llvm_unreachable("unhandled construct"); // LCOV_EXCL_LINE
}

/// What clad emits where the construct is missed: the work the reader pays for.
inline const char* costOf(AnalysisDesc S) {
  switch (S) {
#define CLAD_ANALYSIS_DESC(Id, Analysis, Name, Code, Cost)                     \
  case AnalysisDesc::Id:                                                       \
    return Cost;
#define CLAD_ANALYSIS_MISS(Id, Desc, Detail)
#include "AnalysisDescs.def"
  }
  llvm_unreachable("unhandled construct"); // LCOV_EXCL_LINE
}

/// The number a report prints for the construct, as CLAD1001 and up. A reader
/// searches for it and a build log keeps it, so it never changes; it is also
/// the anchor of the construct's section in the user documentation.
inline unsigned codeOf(AnalysisDesc S) {
  switch (S) {
#define CLAD_ANALYSIS_DESC(Id, Analysis, Name, Code, Cost)                     \
  case AnalysisDesc::Id:                                                       \
    return Code;
#define CLAD_ANALYSIS_MISS(Id, Desc, Detail)
#include "AnalysisDescs.def"
  }
  llvm_unreachable("unhandled construct"); // LCOV_EXCL_LINE
}

/// The construct the miss is about. Only a miss that happened has one.
inline AnalysisDesc descOf(AnalysisMiss M) {
  assert(M != AnalysisMiss::None && "a miss that is not one has no construct");
  switch (M) {
  case AnalysisMiss::None: // LCOV_EXCL_LINE: ruled out above
    break;                 // LCOV_EXCL_LINE
#define CLAD_ANALYSIS_DESC(Id, Analysis, Name, Code, Cost)
#define CLAD_ANALYSIS_MISS(Id, Desc, Detail)                                   \
  case AnalysisMiss::Id:                                                       \
    return AnalysisDesc::Desc;
#include "AnalysisDescs.def"
  }
  llvm_unreachable("unhandled miss"); // LCOV_EXCL_LINE
}

/// What the input did instead, in one sentence. Only a miss that happened
/// has one: nothing files a miss without a reason.
inline const char* detailOf(AnalysisMiss M) {
  assert(M != AnalysisMiss::None && "a miss that is not one has no detail");
  switch (M) {
  case AnalysisMiss::None: // LCOV_EXCL_LINE: ruled out above
    break;                 // LCOV_EXCL_LINE
#define CLAD_ANALYSIS_DESC(Id, Analysis, Name, Code, Cost)
#define CLAD_ANALYSIS_MISS(Id, Desc, Detail)                                   \
  case AnalysisMiss::Id:                                                       \
    return Detail;
#include "AnalysisDescs.def"
  }
  llvm_unreachable("unhandled miss"); // LCOV_EXCL_LINE
}

/// Which analysis acts on the construct.
inline AnalysisId analysisOf(AnalysisDesc S) {
  switch (S) {
#define CLAD_ANALYSIS_DESC(Id, Analysis, Name, Code, Cost)                     \
  case AnalysisDesc::Id:                                                       \
    return AnalysisId::Analysis;
#define CLAD_ANALYSIS_MISS(Id, Desc, Detail)
#include "AnalysisDescs.def"
  }
  llvm_unreachable("unhandled construct"); // LCOV_EXCL_LINE
}

/// One construct an analysis looked for and did not find, filed by an analysis
/// that keeps no result of its own to hang it on. The construct follows from
/// the miss, so only the miss is stored.
struct AnalysisMissRecord {
  AnalysisMiss Why = AnalysisMiss::None;
  /// The token that missed, which is also where the cost of missing it lands.
  clang::SourceLocation At;

  bool operator==(const AnalysisMissRecord& O) const {
    return Why == O.Why && At == O.At;
  }
};

/// The misses filed against one function, kept behind a pointer so the
/// request that owns them does not carry this header.
struct AnalysisMisses {
  llvm::SmallVector<AnalysisMissRecord, 0> Records;
};

/// The answer of an analysis that was asked to prove something: the fact, or
/// the construct the input missed and where it missed it.
///
/// An analysis that declines has to say what would have to change and point
/// at it, so the only constructor that can refuse demands both. Reading the
/// verdict is not optional either: dropping one unread trips an assertion.
///
/// Copy-only: a verdict is two scalars and a location, so a move would copy
/// the same bytes, and both would hand the obligation to read over the same
/// way. Nothing is lost by not having one.
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
template <class T> class [[nodiscard]] Proven {
  T m_Fact{};
  AnalysisMiss m_Why = AnalysisMiss::None;
  clang::SourceLocation m_At;
#ifndef NDEBUG
  mutable bool m_Read = false;
#endif

  void markRead() const {
#ifndef NDEBUG
    m_Read = true;
#endif
  }

public:
  /// The fact, proven.
  Proven(T Fact) : m_Fact(std::move(Fact)) {} // NOLINT(*-explicit-constructor)

  /// Not proven: which construct \p M the input missed, and the token \p At
  /// that missed it -- the one a diagnostic puts its caret on, in the user's
  /// own code rather than in anything clad generated.
  static Proven miss(AnalysisMiss M, clang::SourceLocation At) {
    assert(M != AnalysisMiss::None && "a miss has to say which one");
    assert(At.isValid() && "a miss has to point at the code that missed");
    Proven P{};
    P.m_Why = M;
    P.m_At = At;
    return P;
  }

  Proven(const Proven& O) { *this = O; }
  Proven& operator=(const Proven& O) {
    m_Fact = O.m_Fact;
    m_Why = O.m_Why;
    m_At = O.m_At;
#ifndef NDEBUG
    m_Read = false; // the copy carries the obligation now
#endif
    O.markRead();
    return *this;
  }
  ~Proven() {
    assert(m_Read && "an analysis verdict was dropped without being read");
  }

  explicit operator bool() const {
    markRead();
    return m_Why == AnalysisMiss::None;
  }
  const T& operator*() const {
    assert(m_Why == AnalysisMiss::None && "read of a fact that was not proven");
    markRead();
    return m_Fact;
  }
  const T* operator->() const { return &**this; }

  /// Which construct the input missed. Only meaningful where the verdict is not
  /// proven, which is what the caller has just checked.
  AnalysisMiss why() const {
    assert(m_Why != AnalysisMiss::None && "read of a miss that did not happen");
    markRead();
    return m_Why;
  }
  clang::SourceLocation where() const {
    assert(m_Why != AnalysisMiss::None && "read of a miss that did not happen");
    markRead();
    return m_At;
  }

private:
  Proven() = default;
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_ANALYSES_H
