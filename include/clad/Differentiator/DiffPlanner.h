#include "clang/AST/RecursiveASTVisitor.h"

#include "llvm/ADT/SmallSet.h"

#include "clad/Differentiator/DerivedTypeEssentials.h"

#include <map>
#include <string>
namespace clang {
  class ASTContext;
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

  enum class DiffMode {
    unknown = 0,
    forward,
    reverse,
    hessian,
    jacobian,
    error_estimation
  };

  /// A struct containing information about request to differentiate a function.
  struct DiffRequest {
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
  };

  using DiffSchedule = llvm::SmallVector<DiffRequest, 16>;
  using DiffInterval = std::vector<clang::SourceRange>;
  using DerivativesSet = llvm::SmallSet<const clang::Decl*, 16>;

  class DiffCollector: public clang::RecursiveASTVisitor<DiffCollector> {
    /// The source interval where clad was activated.
    ///
    DiffInterval& m_Interval;

    /// The list of already generated derivatives. There is no need to collect
    /// calls as there are none.
    ///
    const DerivativesSet& m_GeneratedDerivatives;

    /// The diff step-by-step plan for differentiation.
    ///
    DiffSchedule& m_DiffPlans;

    /// If set it means that we need to find the called functions and
    /// add them for implicit diff.
    ///
    const clang::FunctionDecl* m_TopMostFD = nullptr;

    llvm::SmallVector<clang::ClassTemplateSpecializationDecl*, 16>& m_DerivedTypeRequests;
    clang::Sema& m_Sema;

  public:
    DiffCollector(
        clang::DeclGroupRef DGR, DiffInterval& Interval,
        const DerivativesSet& Derivatives, DiffSchedule& plans,
        llvm::SmallVector<clang::ClassTemplateSpecializationDecl*, 16>& derivedTypeRequests,
        clang::Sema& S);
    bool VisitCallExpr(clang::CallExpr* E);
    bool VisitCXXRecordDecl(clang::CXXRecordDecl* RD);
    bool VisitVarDecl(clang::VarDecl* VD);
  private:
    bool isInInterval(clang::SourceLocation Loc) const;
  };
}
