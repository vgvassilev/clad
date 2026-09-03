#ifndef CLAD_DIFF_PLANNER_H
#define CLAD_DIFF_PLANNER_H

#include "clad/Differentiator/DerivedFnCollector.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DynamicGraph.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clad/Differentiator/Timers.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <set>

namespace clang {
class CallExpr;
class CompilerInstance;
class DeclGroupRef;
class Expr;
class FunctionDecl;
class ParmVarDecl;
class Sema;
class Type;
} // namespace clang

namespace clad {
using OwnedAnalysisContexts =
    llvm::SmallVector<std::unique_ptr<clang::AnalysisDeclContext>, 4>;
using ParamSet = std::set<const clang::ParmVarDecl*>;
using ParamInfo = std::map<const clang::FunctionDecl*, ParamSet>;
/// A read-only, AD-oriented view over the primal being differentiated: it
/// wraps the primal FunctionDecl and surfaces the AD-relevant facts the
/// FunctionDecl itself does not. Recording such facts here, rather than
/// rediscovering them inside a visitor, keeps them available to every visitor
/// and correct after a request is copied and re-pointed at another Function.
struct DiffRequest {
  /// Recognises a C heap-memory builtin call and centralises the invariants
  /// reverse mode must preserve for it, so all memory-op reasoning goes through
  /// one place instead of ad-hoc name checks scattered across the code base.
  /// The planner uses it to record which pointers are reallocated in place; the
  /// reverse-mode visitor uses it to emit and undo the resize.
  class AllocCallInfo {
  public:
    enum class Kind : std::uint8_t { None, Malloc, Calloc, Realloc, Free };

    AllocCallInfo() = default;

    // Recognise E as a memory builtin; Kind::None if it is not one. Strips the
    // C-style cast that wraps the call (e.g. `(double*)realloc(...)`), so
    // IgnoreParenCasts, not IgnoreParenImpCasts, is required here.
    [[nodiscard]] static AllocCallInfo recognize(clang::Expr* E) {
      auto* CE = llvm::dyn_cast_or_null<clang::CallExpr>(
          E ? E->IgnoreParenCasts() : nullptr);
      if (!CE)
        return {};
      const clang::FunctionDecl* FD = CE->getDirectCallee();
      // getName() asserts on non-identifier names (operators, constructors),
      // which vector/STL code produces; the builtins are plain identifiers.
      if (!FD || !FD->getDeclName().isIdentifier())
        return {};
      Kind k = llvm::StringSwitch<Kind>(FD->getName())
                   .Case("malloc", Kind::Malloc)
                   .Case("calloc", Kind::Calloc)
                   .Case("realloc", Kind::Realloc)
                   .Case("free", Kind::Free)
                   .Default(Kind::None);
      return AllocCallInfo(k, CE);
    }

    [[nodiscard]] Kind getKind() const { return m_Kind; }
    [[nodiscard]] clang::CallExpr* getCall() const { return m_Call; }

    // The number-of-bytes operand that a following memset must zero:
    // malloc(n) -> n, realloc(p, n) -> n. calloc self-zeroes and needs no
    // memset, so it (and free/none) report null here.
    [[nodiscard]] clang::Expr* memsetByteSize() const {
      switch (m_Kind) {
      case Kind::Malloc:
        return m_Call->getArg(0);
      case Kind::Realloc:
        return m_Call->getArg(1);
      default:
        return nullptr;
      }
    }

    // True for an in-place `p = realloc(p, n)`: the LHS is realloc's own
    // pointer argument. Only then may the reallocated pointer be kept across
    // the call (realloc frees the old block, so a saved pointer would dangle).
    [[nodiscard]] bool isInPlaceRealloc(const clang::Expr* LHS) const {
      if (m_Kind != Kind::Realloc || m_Call->getNumArgs() == 0)
        return false;
      const auto* LDRE =
          llvm::dyn_cast<clang::DeclRefExpr>(LHS->IgnoreParenCasts());
      const auto* ArgDRE = llvm::dyn_cast<clang::DeclRefExpr>(
          m_Call->getArg(0)->IgnoreParenCasts());
      return LDRE && ArgDRE && LDRE->getDecl() == ArgDRE->getDecl();
    }

    // The pointer variable of an in-place `p = realloc(p, n)`, or null.
    [[nodiscard]] const clang::VarDecl*
    getInPlaceReallocPtr(const clang::Expr* LHS) const {
      if (!isInPlaceRealloc(LHS))
        return nullptr;
      return llvm::dyn_cast<clang::VarDecl>(
          llvm::cast<clang::DeclRefExpr>(LHS->IgnoreParenCasts())->getDecl());
    }

  private:
    AllocCallInfo(Kind k, clang::CallExpr* c) : m_Kind(k), m_Call(c) {}
    Kind m_Kind = Kind::None;
    clang::CallExpr* m_Call = nullptr;
  };

private:
  /// Based on To-Be-Recorded analysis performed before differentiation, tells
  /// UsefulToStoreGlobal whether a variable with a given SourceLocation has to
  /// be stored before being changed or not.
  mutable struct TbrRunInfo {
    std::set<const clang::Stmt*> ToBeRecorded;
    ParamInfo m_ModifiedParams;
    ParamInfo m_UsedParams;
    bool HasAnalysisRun = false;
  } m_TbrRunInfo;

  mutable struct ActivityRunInfo {
    std::set<const clang::VarDecl*> VariedDecls;
    std::set<const clang::Stmt*> VariedS;
    /// Whether a call of the primal has a varied argument. Only filled for
    /// requests varied analysis has run on; see shouldHavePushforward().
    llvm::DenseMap<const clang::CallExpr*, bool> VariedCalls;
    bool HasAnalysisRun = false;
  } m_ActivityRunInfo;

  mutable struct UsefulRunInfo {
    std::set<const clang::VarDecl*> UsefulDecls;
    bool HasAnalysisRun = false;
  } m_UsefulRunInfo;

  /// Cache for hasEarlyReturns(): whether the primal body has a return that is
  /// not in tail position. A property of the Function, computed once on demand.
  mutable struct EarlyReturnInfo {
    bool HasEarlyReturns = false;
    bool HasAnalysisRun = false;
  } m_EarlyReturnInfo;

  /// Cache for mayHaveNullTangent(): the primal's pointer variables that may
  /// be given a null tangent. A property of the Function, computed once on
  /// demand.
  mutable struct NullTangentInfo {
    std::set<const clang::VarDecl*> MaybeNullPtrs;
    bool HasAnalysisRun = false;
  } m_NullTangentInfo;

public:
  /// The primal body's tail-position return -- the one an early-return encoder
  /// lets control fall through to, as opposed to an early return that needs a
  /// jump/call. Null when the body does not end in a return. O(1): reads
  /// Function's body directly, so a copied request re-pointed at a new Function
  /// (a lambda's operator(), a pullback callee) answers for its own Function.
  const clang::ReturnStmt* getTailReturn() const;

  /// Whether the primal body has a return that is not the tail return -- one
  /// the reverse mode must encode with the early-return lambda. A fact about
  /// the primal, read directly off Function's body (returns inside nested
  /// lambdas belong to their own function and are skipped). A void function
  /// seeds no return value, so its returns skip the encoding.
  bool hasEarlyReturns() const;

  /// Whether the tangent of the primal's pointer variable \p VD may be null at
  /// run time. Clad cannot synthesize a tangent buffer for a const-qualified
  /// pointer parameter, because its size is unknown, so a pushforward call
  /// receives a null tangent for it, spelling "the derivative of this argument
  /// is identically zero". A pointer of the body derived from such a parameter
  /// -- and one clad has no tangent to give at all -- inherits that, so forward
  /// mode has to guard the reads through their tangents. Every other tangent is
  /// either a buffer clad allocated or a pointer the caller has to provide, and
  /// is therefore never null. Answers false for a non-pointer, and for a
  /// variable belonging to a function other than this request's.
  bool mayHaveNullTangent(const clang::VarDecl* VD) const;

  /// Function to be differentiated.
  const clang::FunctionDecl* Function = nullptr;
  /// Name of the base function to be differentiated. Can be different from
  /// function->getNameAsString() when higher-order derivatives are computed.
  std::string BaseFunctionName = {};
  /// Current derivative order to be computed.
  unsigned CurrentDerivativeOrder = 1;
  /// Highest requested derivative order.
  unsigned RequestedDerivativeOrder = 1;
  /// Context in which the function is being called, or a call to
  /// clad::gradient/differentiate, where function is the first arg.
  clang::Expr* CallContext = nullptr;
  /// Args provided to the call to clad::gradient/differentiate.
  const clang::Expr* Args = nullptr;
  /// Indexes of global GPU args of function as a subset of Args.
  std::vector<size_t> CUDAGlobalArgsIndexes;
  /// Pointer variables that are the target of an in-place `p = realloc(p, n)`
  /// in this function, collected by DiffCollector during planning. Reverse
  /// mode gives exactly these an allocation-size shadow so the realloc can be
  /// undone; other allocated pointers get none. Empty unless the body
  /// reallocates in place.
  std::set<const clang::VarDecl*> InPlaceReallocPtrs;
  /// Whether VD is reallocated in place somewhere in this function.
  bool isInPlaceReallocated(const clang::VarDecl* VD) const {
    return InPlaceReallocPtrs.count(VD) != 0;
  }
  /// Requested differentiation mode, forward or reverse.
  DiffMode Mode = DiffMode::unknown;
  /// If function appears in the call to clad::gradient/differentiate,
  /// the call must be updated and the first arg replaced by the derivative.
  bool CallUpdateRequired = false;
  /// A flag to enable/disable diag warnings/errors during differentiation.
  bool VerboseDiags = false;
  /// Whether each analysis runs for this request. One member per entry in
  /// Analyses.def, spelled Enable<Id>Analysis.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  bool Enable##Id##Analysis = false;
#include "clad/Differentiator/Analyses.def"

  /// Run the same analyses \p Other runs. A derived request -- the pullback of
  /// a callee, the pushforward of a nested call -- covers a different
  /// function, but the user asked for the analyses once, for the whole
  /// differentiation.
  void inheritAnalysesFrom(const DiffRequest& Other) {
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  Enable##Id##Analysis = Other.Enable##Id##Analysis;
#include "clad/Differentiator/Analyses.def"
  }
  /// A flag to emit porting-hint remarks (-fclad-porting-hints) when a function
  /// defined outside the main source file is differentiated by cloning its
  /// definition. Diagnostic-only; it does not affect the generated derivative
  /// and is therefore excluded from request equality.
  bool EmitPortingHints = false;
  /// A flag to request a clad::restore_tracker parameter in the generated
  /// _reverse_forw function.
  bool UseRestoreTracker = false;
  /// A flag specifying whether this differentiation is to be used
  /// in immediate contexts.
  bool ImmediateMode = false;
  /// A flag specifying whether this differentiation is to be used
  /// for error estimation.
  bool EnableErrorEstimation = false;
  /// Assemble the hessian from hessian-vector products -- one pushforward of
  /// the function and one pullback of it -- instead of one second-order
  /// function per direction. The planner decides this (and schedules the
  /// pushforward); HessianModeVisitor only consumes the decision. Derived
  /// from Function, Mode and Args, so excluded from request equality.
  bool UseHessianVectorProducts = false;
  /// Puts the derived function and its code in the diff call
  void updateCall(clang::FunctionDecl* FD, clang::FunctionDecl* OverloadedFD,
                  clang::Sema& SemaRef);
  /// Functor type to be differentiated, if any.
  ///
  /// It is required because we cannot always determine if we are
  /// differentiating a call operator using the function to be
  /// differentiated, for example, when we are computing higher
  /// order derivatives.
  const clang::CXXRecordDecl* Functor = nullptr;
  /// Stores loop checkpoint pragma locations and attached loop locations.
  /// Key: pragma location; value: attached loop location, if any.
  /// The key order is reversed to simplify location range lookups.
  mutable std::map<clang::SourceLocation, clang::SourceLocation, std::greater<>>
      m_CladLoopCheckpoints;

  /// Global VarDecl to differentiate, if any.
  ///
  /// DiffRequests are also used to differentiate global variables.
  const clang::VarDecl* Global = nullptr;

  /// Stores differentiation parameters information. Stored information
  /// includes info on indices range for array parameters, and nested data
  /// member information for record (class) type parameters.
  DiffInputVarsInfo DVI;

  // A flag to enable the use of enzyme for backend instead of clad
  bool use_enzyme = false;

  /// UnresolvedLookupExpr or DeclRefExpr representing the custom derivative
  /// overload
  clang::Expr* CustomDerivative = nullptr;

  /// The trailing clad::pullback_state<S> parameter a custom reverse_forw /
  /// pullback carries, already in argument form (by reference for a
  /// reverse_forw, by value for a pullback), or null when none. clad does not
  /// synthesize this parameter, so it is appended to the expected derivative
  /// signature when matching the overload and when building its overload call.
  clang::QualType PullbackStateParam;
  // FIXME: First per-request fact communicated across the request graph. The
  // finer call-site activity that would let a pullback early-exit inactive
  // branches or narrow toward a directional call wants the same channel: rework
  // DVI and the varied/useful structs into one composable, finer-grained
  // activity model on the request (on the AST), propagated down each subgraph.

  /// A pointer to keep track of the prototype of the derived functions.
  /// For higher order derivatives, we store the entire sequence of
  /// prototypes declared for all orders of derivatives.
  /// This will be useful for forward declaration of the derived functions.
  llvm::SmallVector<clang::FunctionDecl*, 2> DerivedFDPrototypes;

  /// A boolean to indicate if only the declaration of the derived function
  /// is required (and not the definition or body).
  /// This will be particularly useful for pushforward and pullback functions.
  bool DeclarationOnly = false;

  clang::AnalysisDeclContext* m_AnalysisDC = nullptr;

  /// Recomputes `DiffInputVarsInfo` using the current values of data members.
  ///
  /// Differentiation parameters info is computed by parsing the argument
  /// expression for the clad differentiation function calls. The argument is
  /// used to specify independent parameter(s) for differentiation. There are
  /// three valid options for the argument expression:
  ///   1) A string literal, containing comma-separated names of function's
  ///      parameters, as defined in function's definition. If any of the
  ///      parameters are of array or pointer type the indexes of the array
  ///      that needs to be differentiated can also be specified, e.g.
  ///      "arr[1]" or "arr[2:5]". The function will be differentiated w.r.t.
  ///      all the specified parameters.
  ///   2) A numeric literal. The function will be differentiated w.r.t. to
  ///      the parameter corresponding to literal's value index.
  ///   3) If no argument is provided, a default argument is used. The
  ///      function will be differentiated w.r.t. to its every parameter.
  void UpdateDiffParamsInfo(clang::Sema& semaRef);

  /// The pushforward request a hessian assembled from hessian-vector products
  /// consumes. One factory serves both sides: the planner schedules the
  /// request this builds, and HessianModeVisitor builds it again to find that
  /// scheduled derivative, so the two must stay structurally identical.
  [[nodiscard]] DiffRequest
  pushforwardRequestForHessian(clang::Sema& semaRef) const;

  /// Allow comparing DiffRequests.
  bool operator==(const DiffRequest& other) const {
    // Note that CallContext is always different and we should ignore it.
    // CustomDerivative is an Expr* and is not always equal even if
    // the set of overloads is the same.
    // Including AnalysisDC would complicate constructing requests to find the
    // existing once.
    return Function == other.Function &&
           BaseFunctionName == other.BaseFunctionName &&
           CurrentDerivativeOrder == other.CurrentDerivativeOrder &&
           RequestedDerivativeOrder == other.RequestedDerivativeOrder &&
           Args == other.Args && Mode == other.Mode &&
           EnableTBRAnalysis == other.EnableTBRAnalysis &&
           EnableVariedAnalysis == other.EnableVariedAnalysis &&
           EnableUsefulAnalysis == other.EnableUsefulAnalysis &&
           DVI == other.DVI && use_enzyme == other.use_enzyme &&
           DeclarationOnly == other.DeclarationOnly && Global == other.Global &&
           CUDAGlobalArgsIndexes == other.CUDAGlobalArgsIndexes;
  }

  const clang::FunctionDecl* operator->() const { return Function; }

  operator std::string() const {
    std::string res;
    llvm::raw_string_ostream s(res);
    print(s);
    s.flush();
    return res;
  }
  void print(llvm::raw_ostream& Out) const;
  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }

  bool shouldBeRecorded(const clang::Stmt* S) const;
  bool shouldHaveAdjoint(const clang::Stmt* S) const;
  bool shouldHaveAdjoint(const clang::VarDecl* VD) const;
  bool shouldHaveAdjointForw(const clang::VarDecl* VD) const;
  bool isVaried(const clang::Expr* E) const;
  /// Whether forward mode has to differentiate through the call \p CE. False
  /// only when varied analysis proved that no argument of the call depends on
  /// the direction this request differentiates along. A call the analysis
  /// never saw answers true.
  [[nodiscard]] bool shouldHavePushforward(const clang::CallExpr* CE) const;
  std::string ComputeDerivativeName() const;
  bool HasIndependentParameter(const clang::ParmVarDecl* PVD) const;

  std::set<const clang::Stmt*>& getToBeRecorded() const {
    m_TbrRunInfo.HasAnalysisRun = true;
    return m_TbrRunInfo.ToBeRecorded;
  }
  ParamInfo& getModifiedParams() const { return m_TbrRunInfo.m_ModifiedParams; }
  void addFunctionModifiedParams(const clang::FunctionDecl* FD,
                                 const ParamSet& params) {
    m_TbrRunInfo.m_ModifiedParams[FD] = params;
  }
  ParamInfo& getUsedParams() const { return m_TbrRunInfo.m_UsedParams; }
  void addFunctionUsedParams(const clang::FunctionDecl* FD,
                             const ParamSet& params) {
    m_TbrRunInfo.m_UsedParams[FD] = params;
  }
  void addVariedDecl(const clang::VarDecl* init) {
    m_ActivityRunInfo.VariedDecls.insert(init);
  }

  /// Records whether varied analysis found a varied argument of the call
  /// \p CE. Only a forward-mode request is analyzed along exactly the
  /// direction it differentiates; other -enable-va runs are unseeded or
  /// seeded for a caller, and their verdicts must not drop pushforwards.
  /// Monotone, because a loop body is passed over more than once and a later
  /// pass may find an argument varied.
  void recordCallActivity(const clang::CallExpr* CE, bool isVaried) const {
    if (Mode == DiffMode::forward)
      m_ActivityRunInfo.VariedCalls[CE] |= isVaried;
  }

  /// Drops the result of a previous varied-analysis run. A request reused for
  /// another direction -- as the hessian reuses one forward request per row --
  /// has to be analyzed from scratch.
  void resetActivityInfo() const { m_ActivityRunInfo = ActivityRunInfo(); }
  std::set<const clang::VarDecl*>& getVariedDecls() const {
    return m_ActivityRunInfo.VariedDecls;
  }

  std::set<const clang::Stmt*>& getVariedStmt() const {
    return m_ActivityRunInfo.VariedS;
  }

  void addUsefulDecl(const clang::VarDecl* init) {
    m_UsefulRunInfo.UsefulDecls.insert(init);
  }
  std::set<const clang::VarDecl*>& getUsefulDecls() const {
    return m_UsefulRunInfo.UsefulDecls;
  }
  bool HasTbrAnalysisRun() const { return m_TbrRunInfo.HasAnalysisRun; }
};

using DiffInterval = std::vector<clang::SourceRange>;

// FIXME: These are translation-unit-wide defaults taken from the compiler
// invocation, not the options of a request; rename to InvocationOptions.
struct RequestOptions {
  /// Whether each analysis runs, once the switches on the command line have
  /// been resolved against the defaults in Analyses.def.
#define CLAD_ANALYSIS(Id, Name, Legacy, Default, Desc)                         \
  bool Enable##Id##Analysis = Default;
#include "clad/Differentiator/Analyses.def"
  bool EmitPortingHints = false;
};

  class DiffCollector: public clang::RecursiveASTVisitor<DiffCollector> {
    /// The source interval where clad was activated.
    ///
    DiffInterval& m_Interval;

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

    const RequestOptions& m_Options;

    llvm::DenseSet<const clang::FunctionDecl*> m_Traversed;

    bool m_IsTraversingTopLevelDecl = true;

    /// True while Walk is traversing a DeclGroupRef. A traversal can trigger
    /// name lookups that make the ASTReader deserialize pending module decls
    /// and hand them to the consumers, re-entering Walk; such groups are
    /// parked in m_DeferredDGRs and traversed once the active walk finishes.
    bool m_TraversalInFlight = false;

    /// Decl groups delivered while a traversal was in flight (see
    /// m_TraversalInFlight); drained at the end of the outermost Walk.
    llvm::SmallVector<clang::DeclGroupRef, 4> m_DeferredDGRs;

  public:
    DiffCollector(DiffInterval& Interval,
                  clad::DynamicGraph<DiffRequest>& requestGraph, clang::Sema& S,
                  RequestOptions& opts, OwnedAnalysisContexts& AllAnalysisDC);
    /// Run the static planning pass over a group of top-level declarations,
    /// populating the request graph. A no-op when the clad-enabled interval is
    /// empty. Re-entrant calls (e.g. module decls deserialized during a
    /// lookup issued by an active traversal) defer their group until the
    /// active traversal finishes.
    void Walk(clang::DeclGroupRef DGR);
    /// True while Walk is traversing; see m_TraversalInFlight.
    [[nodiscard]] bool isTraversalInFlight() const {
      return m_TraversalInFlight;
    }
    bool VisitCallExpr(clang::CallExpr* E);
    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
    /// Record an in-place `p = realloc(p, n)` on the request whose body is
    /// being traversed, so reverse mode knows p needs an allocation-size
    /// shadow.
    bool VisitBinaryOperator(clang::BinaryOperator* BO);
    bool VisitCXXConstructExpr(clang::CXXConstructExpr* e);
    bool shouldVisitImplicitCode() const { return true; }
    /// Here we use TraverseLambdaExpr and not VisitLambdaExpr to ensure the
    /// new nested DiffRequest is created before the visitor goes to the capture
    /// or constructor initializers. If we use Visit they would be processed
    /// under the parent DiffRequest which is not in the lambda scope.
    bool TraverseLambdaExpr(clang::LambdaExpr* LE);
    /// Plan a lazily-scheduled nested request the static TU walk never reaches
    /// (built in DerivativeBuilder::HandleNestedDiffRequest). Currently records
    /// its early-return flag by walking the request's own body.
    bool PlanNestedRequest(DiffRequest& request);
    bool TraverseFunctionDeclOnce(const clang::FunctionDecl* FD) {
      llvm::SaveAndRestore<bool> Saved(m_IsTraversingTopLevelDecl, false);
      if (m_Traversed.count(FD))
        return true;
      m_Traversed.insert(FD);
      TimedAnalysisRegion R(FD->getNameAsString());
      return TraverseDecl(const_cast<clang::FunctionDecl*>(FD));
    }
    /// Looks up if the user has defined a custom derivative for the given
    /// derivative function. If found, it is automatically attached to the
    /// request in derived function collector.
    /// \param[in] request The request for the derivative to lookup.
    /// \returns true if a custom derivative was found, false otherwise
    bool LookupCustomDerivativeDecl(DiffRequest& request);

  private:
    bool isInInterval(clang::SourceLocation Loc) const;
  };
}

// Define the hash function for DiffRequest.
template <> struct std::hash<clad::DiffRequest> {
    std::size_t operator()(const clad::DiffRequest& DR) const {
      const clang::Decl* D = nullptr;
      if (DR.Function)
        D = DR.Function;
      else
        D = DR.Global;
      // Use the function pointer as the hash of the DiffRequest, it
      // is sufficient to break a reasonable number of collisions.
      if (D->getPreviousDecl())
        return std::hash<const void*>{}(D->getPreviousDecl());
      return std::hash<const void*>{}(D);
    }
};

#endif
