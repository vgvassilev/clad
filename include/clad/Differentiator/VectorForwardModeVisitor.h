#ifndef CLAD_VECTOR_FORWARD_MODE_VISITOR_H
#define CLAD_VECTOR_FORWARD_MODE_VISITOR_H

#include "BaseForwardModeVisitor.h"
#include "DerivativeBuilder.h"

#include "clang/AST/Expr.h"

#include <cstddef>
#include <unordered_map>

namespace clad {
/// A visitor for processing the function code in vector forward mode.
/// Used to compute derivatives by clad::vector_forward_differentiate.
class VectorForwardModeVisitor : public BaseForwardModeVisitor {
protected:
  /// Tracks where the next independent variable starts in the flat vector of
  /// all of them: the sizes of the array parameters seen so far plus the
  /// number of scalar ones. Array sizes are only known at run time, so that
  /// half is an accumulated expression; the scalar half is a counter folded
  /// into a literal on demand.
  ///
  /// The tracker clones every expression it takes in and hands out. The sum
  /// keeps growing after an offset is taken, and each offset goes into a
  /// different part of the derivative; every AST node has exactly one parent.
  class IndVarOffsetTracker {
    VectorForwardModeVisitor* m_V;
    /// Sum of the sizes of the array parameters seen so far, null until the
    /// first one.
    clang::Expr* m_ArrayCount = nullptr;
    /// Number of scalar parameters seen so far.
    std::size_t m_ScalarCount = 0;

  public:
    explicit IndVarOffsetTracker(VectorForwardModeVisitor& V) : m_V(&V) {}
    /// Build a fresh expression for the offset of the current parameter.
    [[nodiscard]] clang::Expr* buildOffset() const;
    /// Account for an array parameter of the given size.
    void advanceByArray(const clang::Expr* size);
    /// Account for a scalar parameter.
    void advanceByScalar() { ++m_ScalarCount; }
  };

  llvm::SmallVector<const clang::ValueDecl*, 16> m_IndependentVars;
  /// Map used to keep track of parameter variables w.r.t which the
  /// the derivative is being computed. This is separate from the
  /// m_Variables map because all other intermediate variables will have
  /// derivatives as vectors.
  std::unordered_map<const clang::ValueDecl*, clang::Expr*> m_ParamVariables;
  /// The generated `indepVarCount` variable (total number of independent
  /// variables). Cached as the decl so each read rebuilds a fresh DeclRef
  /// instead of sharing one node.
  clang::VarDecl* m_IndVarCountDecl = nullptr;

  /// Build a fresh reference to the independent-variable-count variable.
  clang::Expr* buildIndVarCountRef() { return BuildDeclRef(m_IndVarCountDecl); }

public:
  VectorForwardModeVisitor(DerivativeBuilder& builder,
                           const DiffRequest& request);
  ~VectorForwardModeVisitor();

  ///\brief Produces the first derivative of a given function with
  /// respect to multiple parameters.
  ///
  ///\returns The differentiated and potentially created enclosing
  /// context.
  ///
  DerivativeAndOverload Derive() override;

  /// Builds and returns the sequence of derived function parameters for
  //  vectorized forward mode.
  ///
  /// Information about the original function, derived function, derived
  /// function parameter types and the differentiation mode are implicitly
  /// taken from the data member variables.
  llvm::SmallVector<clang::ParmVarDecl*, 8>
  BuildVectorModeParams(DiffParams& diffParams, clang::Expr*& indVarCountExpr);

  /// Get an expression used to initialize the one-hot vector for the
  /// given index and size. A one-hot vector is a vector with all elements
  /// set to 0 except for one element which is set to 1.
  ///
  /// For example: for index = 2 and size = 4, the returned expression
  /// is: {0, 0, 1, 0}
  clang::Expr* getOneHotInitExpr(size_t index, size_t size,
                                 clang::QualType type);

  /// Get an expression used to initialize a zero vector of the given size.
  ///
  /// For example: for size = 4, the returned expression is: {0, 0, 0, 0}
  clang::Expr* getZeroInitListExpr(size_t size, clang::QualType type);

  StmtDiff VisitFloatingLiteral(const clang::FloatingLiteral* FL) override;
  StmtDiff VisitIntegerLiteral(const clang::IntegerLiteral* IL) override;
  StmtDiff
  VisitArraySubscriptExpr(const clang::ArraySubscriptExpr* ASE) override;
  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
  // Decl is not Stmt, so it cannot be visited directly.
  DeclDiff<clang::VarDecl>
  DifferentiateVarDecl(const clang::VarDecl* VD) override;

  std::string GetPushForwardFunctionSuffix() override;
  DiffMode GetPushForwardMode() override;
};
} // end namespace clad

#endif // CLAD_VECTOR_FORWARD_MODE_VISITOR_H
