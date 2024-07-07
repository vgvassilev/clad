#ifndef CLAD_DIFF_PLANNER_H
#define CLAD_DIFF_PLANNER_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/SmallSet.h"
#include "clad/Differentiator/DiffMode.h"
#include "clad/Differentiator/DynamicGraph.h"
#include "clad/Differentiator/ParseDiffArgsTypes.h"

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
  clang::CallExpr* CallContext = nullptr;
  /// Args provided to the call to clad::gradient/differentiate.
  const clang::Expr* Args = nullptr;
  /// Requested differentiation mode, forward or reverse.
  DiffMode Mode = DiffMode::unknown;
  /// If function appears in the call to clad::gradient/differentiate,
  /// the call must be updated and the first arg replaced by the derivative.
  bool CallUpdateRequired = false;
  /// A flag to enable/disable diag warnings/errors during differentiation.
  bool VerboseDiags = false;
  /// A flag to enable TBR analysis during reverse-mode differentiation.
  bool EnableTBRAnalysis = false;
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

  /// Define the == operator for DiffRequest.
  bool operator==(const DiffRequest& other) const {
    // either function match or previous declaration match
    return Function == other.Function &&
           BaseFunctionName == other.BaseFunctionName &&
           CurrentDerivativeOrder == other.CurrentDerivativeOrder &&
           RequestedDerivativeOrder == other.RequestedDerivativeOrder &&
           CallContext == other.CallContext && Args == other.Args &&
           Mode == other.Mode && EnableTBRAnalysis == other.EnableTBRAnalysis &&
           DVI == other.DVI && use_enzyme == other.use_enzyme &&
           DeclarationOnly == other.DeclarationOnly;
  }

  const clang::FunctionDecl* operator->() const { return Function; }

  // String operator for printing the node.
  operator std::string() const {
    std::string res = BaseFunctionName + "__order_" +
                      std::to_string(CurrentDerivativeOrder) + "__mode_" +
                      DiffModeToString(Mode);
    if (EnableTBRAnalysis)
      res += "__TBR";
    return res;
  }

  bool shouldBeRecorded(clang::Expr* E) const;
};

  using DiffInterval = std::vector<clang::SourceRange>;

  struct RequestOptions {
    /// This is a flag to indicate the default behaviour to enable/disable
    /// TBR analysis during reverse-mode differentiation.
    bool EnableTBRAnalysis = false;
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
    const clang::FunctionDecl* m_TopMostFD = nullptr;
    clang::Sema& m_Sema;

    RequestOptions& m_Options;

  public:
    DiffCollector(clang::DeclGroupRef DGR, DiffInterval& Interval,
                  clad::DynamicGraph<DiffRequest>& requestGraph, clang::Sema& S,
                  RequestOptions& opts);
    bool VisitCallExpr(clang::CallExpr* E);

  private:
    bool isInInterval(clang::SourceLocation Loc) const;
  };
}

// Define the hash function for DiffRequest.
template <> struct std::hash<clad::DiffRequest> {
    std::size_t operator()(const clad::DiffRequest& DR) const {
      // Use the function pointer as the hash of the DiffRequest, it
      // is sufficient to break a reasonable number of collisions.
      if (DR.Function->getPreviousDecl())
        return std::hash<const void*>{}(DR.Function->getPreviousDecl());
      return std::hash<const void*>{}(DR.Function);
    }
};

#endif
