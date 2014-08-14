#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
  class ASTContext;
  class CallExpr;
  class CompilerInstance;
  class DeclGroupRef;
  class Expr;
  class FunctionDecl;
  class ParmVarDecl;
  class Sema;
}

namespace clad {

  ///\brief A pair function, independent variable.
  ///
  class FunctionDeclInfo {
  private:
    clang::FunctionDecl* m_FD;
    clang::ParmVarDecl* m_PVD;
  public:
    FunctionDeclInfo(clang::FunctionDecl* FD, clang::ParmVarDecl* PVD);
    clang::FunctionDecl* getFD() const { return m_FD; }
    clang::ParmVarDecl* getPVD() const { return m_PVD; }
    bool isValid() const { return m_FD && m_PVD; }
    LLVM_DUMP_METHOD void dump() const;
  };

  ///\brief The list of the dependent functions which also need differentiation
  /// because they are called by the function we are asked to differentitate.
  ///
  class DiffPlan {
  private:
    typedef llvm::SmallVector<FunctionDeclInfo, 16> Functions;
    Functions m_Functions;
    clang::CallExpr* m_CallToUpdate;
    unsigned m_RequestedDerivativeOrder;
    unsigned m_CurrentDerivativeOrder;
    unsigned m_ArgIndex;
  public:
    DiffPlan() : m_CallToUpdate(0), m_RequestedDerivativeOrder(1),
                 m_CurrentDerivativeOrder(1), m_ArgIndex(0) { }
    typedef Functions::iterator iterator;
    typedef Functions::const_iterator const_iterator;
    unsigned getRequestedDerivativeOrder() { return m_RequestedDerivativeOrder;}
    void setCurrentDerivativeOrder(unsigned val) {
      m_CurrentDerivativeOrder = val;
    }
    unsigned getCurrentDerivativeOrder() { return m_CurrentDerivativeOrder;}

    void push_back(FunctionDeclInfo FDI) { m_Functions.push_back(FDI); }
    iterator begin() { return m_Functions.begin(); }
    iterator end() { return m_Functions.end(); }
    const_iterator begin() const { return m_Functions.begin(); }
    const_iterator end() const { return m_Functions.end(); }
    size_t size() const { return m_Functions.size(); }
    void setCallToUpdate(clang::CallExpr* CE) { m_CallToUpdate = CE; }
    void updateCall(clang::FunctionDecl* FD, clang::Sema& SemaRef);
    LLVM_DUMP_METHOD void dump();
    unsigned getArgIndex() { return m_ArgIndex;}
    void setArgIndex(unsigned val) { m_ArgIndex = val; }

    friend class DiffCollector;
  };

  typedef llvm::SmallVector<DiffPlan, 16> DiffPlans;

  class DiffCollector: public clang::RecursiveASTVisitor<DiffCollector> {
  private:
    ///\brief The diff step-by-step plan for differentiation.
    ///
    DiffPlans& m_DiffPlans;

    ///\brief If set it means that we need to find the called functions and
    /// add them for implicit diff.
    ///
    FunctionDeclInfo* m_TopMostFDI;

    clang::Sema& m_Sema;

    DiffPlan& getCurrentPlan() { return m_DiffPlans.back(); }

    ///\brief Tries to find the independent variable of explicitly diffed
    /// functions.
    ///
    clang::ParmVarDecl* getIndependentArg(clang::Expr* argExpr,
                                          clang::FunctionDecl* FD);
  public:
    DiffCollector(clang::DeclGroupRef DGR, DiffPlans& plans, clang::Sema& S);
    void UpdatePlan(clang::FunctionDecl* FD, DiffPlan* plan);
    bool VisitCallExpr(clang::CallExpr* E);
  };
}
