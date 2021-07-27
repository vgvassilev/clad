//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// version: $Id: ClangPlugin.cpp 7 2013-06-01 22:48:03Z v.g.vassilev@gmail.com $
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#ifndef CLAD_DERIVATIVE_BUILDER_H
#define CLAD_DERIVATIVE_BUILDER_H

#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "clang/Sema/Sema.h"
#include "Compatibility.h"

#include <array>
#include <stack>
#include <unordered_map>

namespace clang {
  class ASTContext;
  class CXXOperatorCallExpr;
  class DeclRefExpr;
  class FunctionDecl;
  class MemberExpr;
  class NamespaceDecl;
  class Scope;
  class Sema;
  class Stmt;
}

namespace clad {
  namespace utils {
    class StmtClone;
  }
  struct DiffRequest;
  namespace plugin {
    class CladPlugin;
    clang::FunctionDecl* ProcessDiffRequest(CladPlugin& P, DiffRequest& request);
  }

  struct IndexInterval {
    size_t Start;
    size_t Finish;

    IndexInterval() : Start(0), Finish(0) {}

    IndexInterval(size_t first, size_t last) : Start(first), Finish(last + 1) {}

    IndexInterval(size_t index) : Start(index), Finish(index + 1) {}

    size_t size() {
      return Finish - Start;
    }

    bool isInInterval(size_t n) {
      return n >= Start && n <= Finish;
    }
  };
}

namespace clad {
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
  using DiffParams = llvm::SmallVector<const clang::VarDecl*, 16>;
  using IndexIntervalTable = llvm::SmallVector<IndexInterval, 16>;
  using DiffParamsWithIndices = std::pair<DiffParams, IndexIntervalTable>;

  using VectorOutputs =
      std::vector<std::unordered_map<const clang::VarDecl*, clang::Expr*>>;

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
    std::unique_ptr<utils::StmtClone> m_NodeCloner;
    clang::NamespaceDecl* m_BuiltinDerivativesNSD;
    DeclWithContext cloneFunction(const clang::FunctionDecl* FD,
                                  clad::VisitorBase VB,
                                  clang::DeclContext* DC,
                                  clang::Sema& m_Sema,
                                  clang::ASTContext& m_Context,
                                  clang::SourceLocation& noLoc,
                                  clang::DeclarationNameInfo name,
                                  clang::QualType functionType);
    clang::Expr* findOverloadedDefinition(clang::DeclarationNameInfo DNI,
                            llvm::SmallVectorImpl<clang::Expr*>& CallArgs);
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
    DerivativeBuilder(clang::Sema& S, plugin::CladPlugin& P);
    ~DerivativeBuilder();

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
  };

} // end namespace clad

#endif // CLAD_DERIVATIVE_BUILDER_H
