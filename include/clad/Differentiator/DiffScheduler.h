//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
//------------------------------------------------------------------------------

#ifndef CLAD_DIFFERENTIATOR_DIFFSCHEDULER_H
#define CLAD_DIFFERENTIATOR_DIFFSCHEDULER_H

#include "clad/Differentiator/DerivedFnCollector.h"
#include "clad/Differentiator/DiffPlanner.h"
#include "clad/Differentiator/DynamicGraph.h"
#include "clad/Differentiator/Options.h"

#include <utility>

namespace clang {
class DeclGroupRef;
class Sema;
} // namespace clang

namespace clad {

/// Owns the differentiation request graph and everything that builds it: the
/// collector, the analysis-context pool and the derived-function map.
/// \ingroup pipeline
class DiffScheduler {
  clang::Sema& m_Sema;
  Options m_Options;
  DiffInterval& m_Interval;
  DynamicGraph<DiffRequest> m_Graph;
  OwnedAnalysisContexts m_AllAnalysisDC;
  DerivedFnCollector m_DFC;
  DiffCollector m_Collector;

public:
  DiffScheduler(clang::Sema& S, Options Opts, DiffInterval& Interval)
      : m_Sema(S), m_Options(std::move(Opts)), m_Interval(Interval),
        m_Collector(m_Interval, m_Graph, m_Sema, m_Options, m_AllAnalysisDC) {}

  DynamicGraph<DiffRequest>& getGraph() { return m_Graph; }
  DerivedFnCollector& getDerivedFns() { return m_DFC; }

  /// Static planning pass over a group of top-level declarations.
  void Plan(clang::DeclGroupRef DGR) { m_Collector.Walk(DGR); }

  /// True while a static planning traversal is on the stack; work that would
  /// process requests (e.g. immediate constexpr differentiation) must not run
  /// re-entrantly while this holds.
  [[nodiscard]] bool isTraversalInFlight() const {
    return m_Collector.isTraversalInFlight();
  }

  /// Plan a single lazily-scheduled request (a nested or higher-order
  /// derivative) that the static walk never reached.
  void Plan(DiffRequest& R) { m_Collector.PlanNestedRequest(R); }

  /// Plans the reverse passes a hessian is built from, now that the
  /// first-order derivatives they differentiate exist (see
  /// DiffRequest::PlansReversePasses).
  void PlanReversePasses(const DiffRequest& Hessian) {
    // The derivatives call templates, such as custom derivatives, that are
    // not instantiated yet; their bodies are walked here.
    m_Sema.PerformPendingInstantiations();
    auto PlanPass = [&](const DiffRequest& First) {
      // A first derivative clad could not build has no reverse pass.
      if (const clang::FunctionDecl* FD = m_DFC.Find(First).DerivedFn()) {
        DiffRequest Pass = Hessian.hessianReversePass(FD);
        Pass.UpdateDiffParamsInfo(m_Sema);
        m_Collector.PlanAsTopMost(Pass);
      }
    };
    if (Hessian.UseHessianVectorProducts)
      PlanPass(Hessian.pushforwardRequestForHessian(m_Sema));
    else
      Hessian.forEachHessianDirection(
          m_Sema, [&](DiffRequest& First, bool) { PlanPass(First); });
  }
};

} // namespace clad

#endif // CLAD_DIFFERENTIATOR_DIFFSCHEDULER_H
