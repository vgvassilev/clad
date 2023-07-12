#ifndef CLAD_VECTOR_FORWARD_MODE_VISITOR_H
#define CLAD_VECTOR_FORWARD_MODE_VISITOR_H

#include "BaseForwardModeVisitor.h"

#include <unordered_map>

namespace clad {
/// A visitor for processing the function code in vector forward mode.
/// Used to compute derivatives by clad::vector_forward_differentiate.
class VectorForwardModeVisitor : public BaseForwardModeVisitor {
private:
  llvm::SmallVector<const clang::ValueDecl*, 16> m_IndependentVars;
  /// Map used to keep track of parameter variables w.r.t which the
  /// the derivative is being computed. This is separate from the
  /// m_Variables map because all other intermediate variables will have
  /// derivatives as vectors.
  std::unordered_map<const clang::ValueDecl*, clang::Expr*> m_ParamVariables;

public:
  VectorForwardModeVisitor(DerivativeBuilder& builder);
  ~VectorForwardModeVisitor();

  ///\brief Produces the first derivative of a given function with
  /// respect to multiple parameters.
  ///
  ///\param[in] FD - the function that will be differentiated.
  ///
  ///\returns The differentiated and potentially created enclosing
  /// context.
  ///
  DerivativeAndOverload DeriveVectorMode(const clang::FunctionDecl* FD,
                                         const DiffRequest& request);

  /// Builds and returns the sequence of derived function parameters for
  //  vectorized forward mode.
  ///
  /// Information about the original function, derived function, derived
  /// function parameter types and the differentiation mode are implicitly
  /// taken from the data member variables. In particular, `m_Function`,
  /// `m_Mode` and `m_Derivative` should be correctly set before using this
  /// function.
  llvm::SmallVector<clang::ParmVarDecl*, 8>
  BuildVectorModeParams(DiffParams& diffParams);

  /// Get an expression used to initialize the one-hot vector for the
  /// given index and size. A one-hot vector is a vector with all elements
  /// set to 0 except for one element which is set to 1.
  ///
  /// For example: for index = 2 and size = 4, the returned expression
  /// is: {0, 0, 1, 0}
  clang::Expr* getOneHotInitExpr(size_t index, size_t size);

  /// Get an expression used to initialize a zero vector of the given size.
  ///
  /// For example: for size = 4, the returned expression is: {0, 0, 0, 0}
  clang::Expr* getZeroInitListExpr(size_t size);

  StmtDiff VisitReturnStmt(const clang::ReturnStmt* RS) override;
  // Decl is not Stmt, so it cannot be visited directly.
  VarDeclDiff DifferentiateVarDecl(const clang::VarDecl* VD) override;
};
} // end namespace clad

#endif // CLAD_VECTOR_FORWARD_MODE_VISITOR_H
