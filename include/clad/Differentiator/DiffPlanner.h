#ifndef CLAD_DIFF_PLANNER_H
#define CLAD_DIFF_PLANNER_H

#include "clad/Differentiator/DerivedFnCollector.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DynamicGraph.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"
#include "clad/Differentiator/Timers.h"

#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/SaveAndRestore.h"
#include "llvm/Support/raw_ostream.h"

#include <iterator>
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

/// A struct containing information about request to differentiate a function.
struct DiffRequest {
private:
  /// Based on To-Be-Recorded analysis performed before differentiation, tells
  /// UsefulToStoreGlobal whether a variable with a given SourceLocation has to
  /// be stored before being changed or not.
  mutable struct TbrRunInfo {
    std::set<clang::SourceLocation> ToBeRecorded;
    bool HasAnalysisRun = false;
  } m_TbrRunInfo;

  mutable struct ActivityRunInfo {
    std::set<const clang::VarDecl*> VariedDecls;
    bool HasAnalysisRun = false;
  } m_ActivityRunInfo;

  mutable struct UsefulRunInfo {
    std::set<const clang::VarDecl*> UsefulDecls;
    bool HasAnalysisRun = false;
  } m_UsefulRunInfo;

public:
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
  /// Requested differentiation mode, forward or reverse.
  DiffMode Mode = DiffMode::unknown;
  /// If function appears in the call to clad::gradient/differentiate,
  /// the call must be updated and the first arg replaced by the derivative.
  bool CallUpdateRequired = false;
  /// A flag to enable/disable diag warnings/errors during differentiation.
  bool VerboseDiags = false;
  /// A flag to enable TBR analysis during reverse-mode differentiation.
  bool EnableTBRAnalysis = false;
  bool EnableVariedAnalysis = false;
  bool EnableUsefulAnalysis = false;
  /// A flag specifying whether this differentiation is to be used
  /// in immediate contexts.
  bool ImmediateMode = false;
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

  /// A pointer to keep track of the prototype of the derived functions.
  /// For higher order derivatives, we store the entire sequence of
  /// prototypes declared for all orders of derivatives.
  /// This will be useful for forward declaration of the derived functions.
  llvm::SmallVector<clang::FunctionDecl*, 2> DerivedFDPrototypes;

  /// A boolean to indicate if only the declaration of the derived function
  /// is required (and not the definition or body).
  /// This will be particularly useful for pushforward and pullback functions.
  bool DeclarationOnly = false;

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

  /// Allow comparing DiffRequests.
  bool operator==(const DiffRequest& other) const {
    // Note that CallContext is always different and we should ignore it.
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

  bool shouldBeRecorded(clang::Expr* E) const;
  bool shouldHaveAdjoint(const clang::VarDecl* VD) const;
  bool shouldHaveAdjointForw(const clang::VarDecl* VD) const;
  bool isVaried(const clang::Expr* E) const;
  std::string ComputeDerivativeName() const;
  bool HasIndependentParameter(const clang::ParmVarDecl* PVD) const;

  void addVariedDecl(const clang::VarDecl* init) {
    m_ActivityRunInfo.VariedDecls.insert(init);
  }
  std::set<const clang::VarDecl*>& getVariedDecls() const {
    return m_ActivityRunInfo.VariedDecls;
  }
  void addUsefulDecl(const clang::VarDecl* init) {
    m_UsefulRunInfo.UsefulDecls.insert(init);
  }
  std::set<const clang::VarDecl*>& getUsefulDecls() const {
    return m_UsefulRunInfo.UsefulDecls;
  }
};

  using DiffInterval = std::vector<clang::SourceRange>;

  struct RequestOptions {
    /// This is a flag to indicate the default behaviour to enable/disable
    /// TBR analysis during reverse-mode differentiation.
    bool EnableTBRAnalysis = false;
    bool EnableVariedAnalysis = false;
    bool EnableUsefulAnalysis = false;
  };

  class DiffCollector: public clang::RecursiveASTVisitor<DiffCollector> {
    /// The source interval where clad was activated.
    ///
    DiffInterval& m_Interval;

    /// Graph to store the dependencies between different requests.
    ///
    clad::DynamicGraph<DiffRequest>& m_DiffRequestGraph;

    /// If set it means that we need to find the called functions and
    /// add them for implicit diff.
    ///
    const DiffRequest* m_TopMostReq = nullptr;

    const DiffRequest* m_ParentReq = nullptr;
    clang::Sema& m_Sema;

    const RequestOptions& m_Options;

    llvm::DenseSet<const clang::FunctionDecl*> m_Traversed;

    bool m_IsTraversingTopLevelDecl = true;

    DerivedFnCollector& m_DFC;

  public:
    DiffCollector(clang::DeclGroupRef DGR, DiffInterval& Interval,
                  clad::DynamicGraph<DiffRequest>& requestGraph, clang::Sema& S,
                  RequestOptions& opts, DerivedFnCollector& DFC);
    bool VisitCallExpr(clang::CallExpr* E);
    bool VisitDeclRefExpr(clang::DeclRefExpr* DRE);
    bool VisitCXXConstructExpr(clang::CXXConstructExpr* e);
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
    bool LookupCustomDerivativeDecl(const DiffRequest& request);

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
