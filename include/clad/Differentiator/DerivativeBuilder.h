//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DERIVATIVE_BUILDER_H
#define CLAD_DERIVATIVE_BUILDER_H

#include "Compatibility.h"
#include "clad/Differentiator/DerivedTypesHandler.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"
#include "clad/Differentiator/DerivedTypeEssentials.h"
#include <array>
#include <map>
#include <stack>
#include <unordered_map>

namespace clang {
  class ASTContext;
  class CXXOperatorCallExpr;
  class CXXRecordDecl;
  class DeclRefExpr;
  class FunctionDecl;
  class MemberExpr;
  class NamespaceDecl;
  class Scope;
  class Sema;
  class Stmt;
} // namespace clang

namespace clad {
  namespace utils {
    class StmtClone;
  }
  struct DiffRequest;
  namespace plugin {
    class CladPlugin;
    clang::FunctionDecl* ProcessDiffRequest(CladPlugin& P,
                                            DiffRequest& request);
  } // namespace plugin

  struct IndexInterval {
    size_t Start;
    size_t Finish;

    IndexInterval() : Start(0), Finish(0) {}

    IndexInterval(size_t first, size_t last) : Start(first), Finish(last + 1) {}

    IndexInterval(size_t index) : Start(index), Finish(index + 1) {}

    size_t size() { return Finish - Start; }

    bool isInInterval(size_t n) { return n >= Start && n <= Finish; }
  };
} // namespace clad

namespace clad {
  class ErrorEstimationHandler;
  class DerivedTypesHandler;
  class FPErrorEstimationModel;
  // A pointer to a the handler to be used for estimation requests.
  extern std::unique_ptr<ErrorEstimationHandler> errorEstHandler;

  /// A pair of FunctionDecl and potential enclosing context, e.g. a function
  /// in nested namespaces.
  // This is the type returned by cloneFunction. Using OverloadedDeclWithContext
  // instead would lead to unnecessarily returning a nullptr in the overloaded
  // FD
  using DeclWithContext = std::pair<clang::FunctionDecl*, clang::Decl*>;
  /// A tuple which consists of a FunctionDecl, it's potential enclosing context
  /// and optionally it's overload FunctionDecl
  using OverloadedDeclWithContext =
      std::tuple<clang::FunctionDecl*, clang::Decl*, clang::FunctionDecl*>;
  using DiffParams = llvm::SmallVector<const clang::ValueDecl*, 16>;
  using IndexIntervalTable = llvm::SmallVector<IndexInterval, 16>;
  using DiffParamsWithIndices = std::pair<DiffParams, IndexIntervalTable>;

  using VectorOutputs =
      std::vector<std::unordered_map<const clang::ValueDecl*, clang::Expr*>>;

  static clang::SourceLocation noLoc{};
  class VisitorBase;
  /// The main builder class which then uses either ForwardModeVisitor or
  /// ReverseModeVisitor based on the required mode.
  class DerivativeBuilder {
  private:
    friend class VisitorBase;
    friend class ForwardModeVisitor;
    friend class ReverseModeVisitor;
    friend class HessianModeVisitor;
    friend class JacobianModeVisitor;

    clang::Sema& m_Sema;
    plugin::CladPlugin& m_CladPlugin;
    clang::ASTContext& m_Context;
    DerivedTypesHandler& m_DTH;
    std::unique_ptr<utils::StmtClone> m_NodeCloner;
    clang::NamespaceDecl* m_BuiltinDerivativesNSD;
    /// A reference to the model to use for error estimation (if any).
    std::unique_ptr<FPErrorEstimationModel> m_EstModel = nullptr;
    clang::NamespaceDecl* m_NumericalDiffNSD;
    /// A flag to keep track of whether error diagnostics are requested by user
    /// for numerical differentiation.
    bool m_PrintNumericalDiffErrorDiag = false;
    /// Stores mapping of derived type names and the corresponding derived
    /// types.
    ///
    /// Derived types are the types that are used to store the derivatives.
    std::map<std::string, clang::QualType> m_DerivedTypes;
    DeclWithContext cloneFunction(const clang::FunctionDecl* FD,
                                  clad::VisitorBase VB, clang::DeclContext* DC,
                                  clang::Sema& m_Sema,
                                  clang::ASTContext& m_Context,
                                  clang::SourceLocation& noLoc,
                                  clang::DeclarationNameInfo name,
                                  clang::QualType functionType);
    /// Looks for a suitable overload for a given function.
    ///
    /// \param[in] DNI The identification information of the function overload
    /// to be found.
    /// \param[in] CallArgs The call args to be used to resolve to the
    /// correct overload.
    /// \param[in] forCustomDerv A flag to keep track of which
    /// namespace we should look in for the overloads.
    /// \param[in] namespaceShouldExist A flag to enforce assertion failure
    /// if the overload function namespace was not found. If false and
    /// the function containing namespace was not found, nullptr is returned.
    ///
    /// \returns The call expression if a suitable function overload was found,
    /// null otherwise.
    clang::Expr*
    findOverloadedDefinition(clang::DeclarationNameInfo DNI,
                             llvm::SmallVectorImpl<clang::Expr*>& CallArgs,
                             bool forCustomDerv = true,
                             bool namespaceShouldExist = true);
    bool noOverloadExists(clang::Expr* UnresolvedLookup,
                          llvm::MutableArrayRef<clang::Expr*> ARargs);
    /// Shorthand to issues a warning or error.
    template <std::size_t N>
    void diag(clang::DiagnosticsEngine::Level level, // Warning or Error
              clang::SourceLocation loc,
              const char (&format)[N],
              llvm::ArrayRef<llvm::StringRef> args = {}) {
      unsigned diagID = m_Sema.Diags.getCustomDiagID(level, format);
      clang::Sema::SemaDiagnosticBuilder stream = m_Sema.Diag(loc, diagID);
      for (auto arg : args)
        stream << arg;
    }

  public:
    DerivativeBuilder(clang::Sema& S, plugin::CladPlugin& P, DerivedTypesHandler& DTH);
    ~DerivativeBuilder();
    /// Reset the model use for error estimation (if any).
    /// \param[in] estModel The error estimation model, can be either
    /// an in-built one (TaylorApprox) or one provided by the user.
    void
    SetErrorEstimationModel(std::unique_ptr<FPErrorEstimationModel> estModel);
    /// Fuction to set the error diagnostic printing value for numerical
    /// differentiation.
    ///
    /// \param[in] \c value The new value to be set.
    void setNumDiffErrDiag(bool value) {
      m_PrintNumericalDiffErrorDiag = value;
    }
    /// Function to return if clad should emit error information for numerical
    /// differentiation.
    ///
    /// \returns The flag  that controls printing of error information for
    /// numerical differentiation.
    bool shouldPrintNumDiffErrs() { return m_PrintNumericalDiffErrorDiag; }
    ///\brief Produces the derivative of a given function
    /// according to a given plan.
    ///
    ///\param[in] FD - the function that will be differentiated.
    ///
    ///\returns The differentiated function and potentially created enclosing
    /// context.
    ///
    OverloadedDeclWithContext Derive(const clang::FunctionDecl* FD,
                                     const DiffRequest& request);
    void AddDerivedType(llvm::StringRef typeName, clang::QualType qType);
    clang::QualType GetDerivedType(llvm::StringRef typeName) const;
  };

} // end namespace clad

#endif // CLAD_DERIVATIVE_BUILDER_H
