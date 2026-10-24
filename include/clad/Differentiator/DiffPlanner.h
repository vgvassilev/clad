#ifndef CLAD_DIFF_PLANNER_H
#define CLAD_DIFF_PLANNER_H

#include "clad/Differentiator/DerivedFnCollector.h"
#include "clad/Differentiator/DiffRequest.h"
#include "clad/Differentiator/DynamicGraph.h"
#include "clad/Differentiator/Timers.h"

#include "clang/AST/Decl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <map>
#include <memory>
#include <set>
#include <vector>

namespace clang {
class BinaryOperator;
class CXXConstructExpr;
class CUDAKernelCallExpr;
class DeclRefExpr;
class LambdaExpr;
class Sema;
} // namespace clang

namespace clad {

using OwnedAnalysisContexts =
    llvm::SmallVector<std::unique_ptr<clang::AnalysisDeclContext>, 4>;

/// Plans derivative dependencies within requested functions.
/// \ingroup pipeline
class DiffCollector : public clang::RecursiveASTVisitor<DiffCollector> {
  /// Graph to store the dependencies between different requests.
  ///
  clad::DynamicGraph<DiffRequest>& m_DiffRequestGraph;
  /// Map that contains all AnalysisDeclContext for all declrations.
  /// Essentially needed for prolonging the lifetime of
  /// unique_ptr<clang::AnalysisDeclContext>.
  OwnedAnalysisContexts& m_AllAnalysisDC;
  /// The contexts PlanNestedRequest built, keyed by the function they
  /// describe, so that repeated planning of the same lazily-scheduled
  /// request reuses one context (and one CFG) instead of building another.
  llvm::DenseMap<const clang::FunctionDecl*, clang::AnalysisDeclContext*>
      m_NestedAnalysisDC;
  /// If set it means that we need to find the called functions and
  /// add them for implicit diff.
  ///
  const DiffRequest* m_TopMostReq = nullptr;

  DiffRequest* m_ParentReq = nullptr;
  clang::Sema& m_Sema;

  /// The requests whose bodies the planner has walked. Two requests for the
  /// same function, such as two pullbacks that take different parameters,
  /// each get a walk: what it records goes to the request walked.
  std::vector<DiffRequest> m_Traversed;

  /// The globals a function's own body reads and assigns, and the functions
  /// it calls directly.
  struct GlobalAccess {
    std::set<const clang::VarDecl*> Reads;
    std::set<const clang::VarDecl*> Writes;
    /// Every global the body assigns, and every global it reads, of any
    /// type a caller can copy aside: what a caller may have to put back
    /// around a replay of it.
    std::set<const clang::VarDecl*> AnyWrites;
    std::set<const clang::VarDecl*> AnyReads;
    std::set<const clang::VarDecl*> IndirectReads;
    std::vector<const clang::FunctionDecl*> Callees;
  };
  /// A std::map: references into it stay valid while it grows.
  std::map<const clang::FunctionDecl*, GlobalAccess> m_GlobalAccess;
  const GlobalAccess& getGlobalAccess(const clang::FunctionDecl* FD);
  /// The union of getGlobalAccess() over the functions reachable from a
  /// callee through direct calls, itself included.
  std::map<const clang::FunctionDecl*, GlobalAccess> m_ReachableGlobalAccess;
  const GlobalAccess&
  getReachableGlobalAccess(const clang::FunctionDecl* Callee);
  /// Whether the call \p E reads a global the differentiated code assigns,
  /// or assigns one it reads; if so, records \p E on the parent request.
  bool touchesActiveGlobal(const clang::Expr* E,
                           const clang::FunctionDecl* Callee);

  /// Whether the body being walked is replaced by a custom derivative.
  bool m_BodyIsReplaced = false;
  /// Reports \p E if it is an active byte copy of floating-point values,
  /// which has no derivative; true if so.
  bool diagnoseByteCopy(const clang::CallExpr* E);

public:
  DiffCollector(clad::DynamicGraph<DiffRequest>& requestGraph, clang::Sema& S,
                OwnedAnalysisContexts& AllAnalysisDC);
  /// Plan a prepared root request.
  bool PlanRequest(DiffRequest& request);
  bool VisitCallExpr(clang::CallExpr* E);
  bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
  /// Record an in-place `p = realloc(p, n)` on the request whose body is
  /// being traversed, so reverse mode knows p needs an allocation-size
  /// shadow.
  bool VisitBinaryOperator(clang::BinaryOperator* BO);
  bool VisitCXXConstructExpr(clang::CXXConstructExpr* e);
  [[nodiscard]] bool shouldVisitImplicitCode() const { return true; }
  /// Here we use TraverseLambdaExpr and not VisitLambdaExpr to ensure the
  /// new nested DiffRequest is created before the visitor goes to the capture
  /// or constructor initializers. If we use Visit they would be processed
  /// under the parent DiffRequest which is not in the lambda scope.
  bool TraverseLambdaExpr(clang::LambdaExpr* LE);
  /// Plans the requests a kernel launch contains without descending into
  /// its execution configuration.
  bool TraverseCUDAKernelCallExpr(clang::CUDAKernelCallExpr* KCE);
  /// Plan a lazily-scheduled nested request the static TU walk never reaches
  /// (built in DerivativeBuilder::HandleNestedDiffRequest). Currently records
  /// its early-return flag by walking the request's own body.
  bool PlanNestedRequest(DiffRequest& request);
  /// Walks the body of the function of \p request, once for each request.
  bool TraverseFunctionDeclOnce(const DiffRequest& request);
  /// Looks up if the user has defined a custom derivative for the given
  /// derivative function. If found, it is automatically attached to the
  /// request in derived function collector.
  /// \param[in] request The request for the derivative to lookup.
  /// \param[out] foundAnyDecl If given, the lookup reports nothing, and is
  /// told whether a declaration of the derivative's name exists at all.
  /// \returns true if a custom derivative was found, false otherwise
  bool LookupCustomDerivativeDecl(DiffRequest& request,
                                  bool* foundAnyDecl = nullptr);

private:
  bool CollectRequest(DiffRequest& request, bool nonDiff = false,
                      bool hasNoMemoryInputForPointerOrRefReturn = false);
};

/// The source ranges where clad was switched on, in order.
using DiffInterval = std::vector<clang::SourceRange>;

/// Owns the differentiation request graph and everything that builds it: the
/// collector, the analysis-context pool and the derived-function map. Filters
/// root invocations by pragma state before passing them to the collector.
/// \ingroup pipeline
class DiffScheduler {
  clang::Sema& m_Sema;
  // Parsing can append intervals, so keep the vector rather than a view.
  const DiffInterval& m_Interval;
  DynamicGraph<DiffRequest> m_Graph;
  OwnedAnalysisContexts m_AllAnalysisDC;
  DerivedFnCollector m_DFC;
  DiffCollector m_Collector;
  /// Roots discovered by re-entrant declaration callbacks wait until the
  /// current root has restored the collector's state.
  bool m_Planning = false;
  llvm::SmallVector<DiffRequest, 4> m_DeferredRequests;

public:
  DiffScheduler(clang::Sema& S, const DiffInterval& Interval)
      : m_Sema(S), m_Interval(Interval),
        m_Collector(m_Graph, S, m_AllAnalysisDC) {}

  DynamicGraph<DiffRequest>& getGraph() { return m_Graph; }
  DerivedFnCollector& getDerivedFns() { return m_DFC; }

  /// Whether a root invocation is enabled by `#pragma clad ON`.
  [[nodiscard]] bool isEnabled(clang::SourceLocation Loc) const;

  /// Derivative generation must wait until root planning is complete.
  [[nodiscard]] bool isPlanning() const { return m_Planning; }

  /// Plan a single lazily-scheduled request (a nested or higher-order
  /// derivative) that the static walk never reached.
  void Plan(DiffRequest& R) { m_Collector.PlanNestedRequest(R); }

  /// Plan a root from an entry-point visitor or an external compiler client.
  /// CallContext must identify the invocation whose location is checked.
  /// If enabled, run the optional Prepare callback synchronously; it returns
  /// false if the request cannot be prepared. Only prepared roots are queued.
  /// Target and dependency traversal is independent of pragma state.
  void PlanRoot(DiffRequest& R, llvm::function_ref<bool()> Prepare = nullptr);
};

} // namespace clad

#endif // CLAD_DIFF_PLANNER_H
