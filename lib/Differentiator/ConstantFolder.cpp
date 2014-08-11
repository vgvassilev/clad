//--------------------------------------------------------------------*- C++ -//
// clad - the C++ Clang-based Automatic Differentiator
//
// A constant folding tool, working on AST level
//
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//----------------------------------------------------------------------------//

#include "ConstantFolder.h"

#include "clang/AST/ASTContext.h"

namespace clad {
  using namespace clang;

  bool evalsToZero(Expr* E, ASTContext& C) {
    llvm::APSInt result;
    if (E->EvaluateAsInt(result, C)) {
      return result.getSExtValue() == 0;
    }

    return false;
  }

  Stmt* ConstantFolder::VisitExpr(Expr* E) {
    llvm::APSInt Val;
    if (E->EvaluateAsInt(Val, m_Context)) {
      return synthesizeLiteral(E->getType(), m_Context, Val.getZExtValue());
    }
    return E;
  }

  Expr* ConstantFolder::fold(Expr* E) {
    if (!m_Enabled)
      return E;
    Stmt* result = Visit(E);
    return cast<Expr>(result);
  }

  Expr* ConstantFolder::synthesizeLiteral(QualType QT, ASTContext& C,
                                          unsigned val) {
    SourceLocation noLoc;
    Expr* Result = 0;
    if (QT->isIntegralType(C)) {
      llvm::APInt APVal(C.getIntWidth(C.IntTy), val);
      Result = IntegerLiteral::Create(C, APVal, C.IntTy, noLoc);
    }
    else {
      llvm::APFloat APVal(C.getFloatTypeSemantics(QT), val);
      Result = FloatingLiteral::Create(C, APVal, /*isexact*/true, QT, noLoc);
    }
    assert(Result && "Must not be zero.");
    return Result;
  }

} // end namespace clad
