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

  enum class DiffMode {
    forward,
    reverse
  };

  ///\brief The list of the dependent functions which also need differentiation
  /// because they are called by the function we are asked to differentitate.
  ///
  class DiffPlan {
  private:
    typedef llvm::SmallVector<clang::FunctionDecl*, 16> Functions;
    Functions m_Functions;
    clang::CallExpr* m_CallToUpdate = nullptr;
    unsigned m_RequestedDerivativeOrder = 1;
    unsigned m_CurrentDerivativeOrder = 1;
    clang::Expr* m_DiffArgs = nullptr;
    DiffMode m_Mode;
  public:
    typedef Functions::iterator iterator;
    typedef Functions::const_iterator const_iterator;

    DiffMode getMode() const {
      return m_Mode;
    }
    void setMode(DiffMode mode) {
      m_Mode = mode;
    }
    unsigned getRequestedDerivativeOrder() const {
      return m_RequestedDerivativeOrder;
   }
    void setCurrentDerivativeOrder(unsigned val) {
      m_CurrentDerivativeOrder = val;
    }
    unsigned getCurrentDerivativeOrder() const {
      return m_CurrentDerivativeOrder;
    }
    void push_back(clang::FunctionDecl* FD) { m_Functions.push_back(FD); }
    iterator begin() { return m_Functions.begin(); }
    iterator end() { return m_Functions.end(); }
    const_iterator begin() const { return m_Functions.begin(); }
    const_iterator end() const { return m_Functions.end(); }
    size_t size() const { return m_Functions.size(); }
    void setCallToUpdate(clang::CallExpr* CE) { m_CallToUpdate = CE; }
    void updateCall(clang::FunctionDecl* FD, clang::Sema& SemaRef);
    clang::Expr* getArgs() const { return m_DiffArgs; }
    LLVM_DUMP_METHOD void dump();

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
    clang::FunctionDecl* m_TopMostFD;

    clang::Sema& m_Sema;

    DiffPlan& getCurrentPlan() { return m_DiffPlans.back(); }

  public:
    DiffCollector(clang::DeclGroupRef DGR, DiffPlans& plans, clang::Sema& S);
    void UpdatePlan(clang::FunctionDecl* FD, DiffPlan* plan);
    bool VisitCallExpr(clang::CallExpr* E);
  };
}
