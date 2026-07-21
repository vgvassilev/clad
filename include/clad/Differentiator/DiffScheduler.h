//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR_DIFFSCHEDULER_H
#define CLAD_DIFFERENTIATOR_DIFFSCHEDULER_H

#include "clad/Differentiator/DerivedFnCollector.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/DynamicGraph.h"

namespace clang {
class DeclGroupRef;
class Sema;
} // namespace clang

namespace clad {

/// Owns the differentiation request graph and everything that builds it: the
/// collector, the analysis-context pool and the derived-function map.
class DiffScheduler {
  clang::Sema& m_Sema;
  RequestOptions m_Options;
  DiffInterval& m_Interval;
  DynamicGraph<DiffRequest> m_Graph;
  OwnedAnalysisContexts m_AllAnalysisDC;
  DerivedFnCollector m_DFC;
  DiffCollector m_Collector;

public:
  DiffScheduler(clang::Sema& S, const RequestOptions& Opts,
                DiffInterval& Interval)
      : m_Sema(S), m_Options(Opts), m_Interval(Interval),
        m_Collector(m_Interval, m_Graph, m_Sema, m_Options, m_AllAnalysisDC) {}

  DynamicGraph<DiffRequest>& getGraph() { return m_Graph; }
  DerivedFnCollector& getDerivedFns() { return m_DFC; }

  /// Static planning pass over a group of top-level declarations.
  void Plan(clang::DeclGroupRef DGR) { m_Collector.Walk(DGR); }

  /// Plan a single lazily-scheduled request (a nested or higher-order
  /// derivative) that the static walk never reached.
  void Plan(DiffRequest& R) { m_Collector.PlanNestedRequest(R); }
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_DIFFSCHEDULER_H
