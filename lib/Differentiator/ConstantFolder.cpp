//--------------------------------------------------------------------*- C++ -//
// clad - the C++ Clang-based Automatic Differentiator
//
// A constant folding tool, working on AST level
//
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//----------------------------------------------------------------------------//

#include "ConstantFolder.h"
#include "clad/Differentiator/Compatibility.h"

#include "clang/AST/ASTContext.h"

namespace clad {
  using namespace clang;

  static bool evalsToN(Expr* E, ASTContext& C, uint64_t N = 0) {
    Expr::EvalResult result;
    if (E->EvaluateAsRValue(result, C)) {
      if (result.Val.isFloat()) {
        using namespace llvm;
        APFloat F = result.Val.getFloat();
        return APFloat(F.getSemantics(), N).compare(F) == APFloat::cmpEqual;
      }
      else if (result.Val.isInt()) {
        return result.Val.getInt().getZExtValue() == N;
      }
    }

    return false;
  }

  static bool evalsToZero(Expr* E, ASTContext& C) {
    return evalsToN(E, C, /*N=*/0);
  }

  static bool evalsToOne(Expr* E, ASTContext& C) {
    return evalsToN(E, C, /*N=*/1);
  }

  static Expr* synthesizeLiteral(QualType QT, ASTContext& C, llvm::APInt val) {
    assert(QT->isIntegralType(C) && "Not an integer type.");
    SourceLocation noLoc;
    return IntegerLiteral::Create(C, val, QT, noLoc);
  }

  static Expr* synthesizeLiteral(QualType QT, ASTContext& C, llvm::APFloat val){
    assert (&C.getFloatTypeSemantics(QT) == &val.getSemantics() && "Mismatch");
    SourceLocation noLoc;
    return FloatingLiteral::Create(C, val, /*isexact*/true, QT, noLoc);
  }

  static Expr* synthesizeLiteral(QualType QT, ASTContext& C, bool val) {
    assert(QT->isBooleanType() && "Not a boolean type.");
    SourceLocation noLoc;
    return new (C) CXXBoolLiteralExpr(val, QT, noLoc);
  }

  static Expr* synthesizeLiteral(QualType QT, ASTContext& C) {
    assert(QT->isPointerType() && "Not a pointer type.");
    SourceLocation noLoc;
    return new (C) CXXNullPtrLiteralExpr(QT, noLoc);
  }

  Expr* ConstantFolder::trivialFold(Expr* E) {
    Expr::EvalResult Result;
    if (E->EvaluateAsRValue(Result, m_Context)) {
      if (Result.Val.isFloat()) {
        llvm::APFloat F = Result.Val.getFloat();
        E = clad::synthesizeLiteral(E->getType(), m_Context, F);
      }
      else if (Result.Val.isInt()) {
        llvm::APSInt I = Result.Val.getInt();
        E = clad::synthesizeLiteral(E->getType(), m_Context, I);
      }
    }
    return E;
  }

  Expr* ConstantFolder::VisitExpr(Expr* E) {
    return E;
  }

  Expr* ConstantFolder::VisitBinaryOperator(BinaryOperator* BinOp) {
    Expr* LHS = cast<Expr>(Visit(BinOp->getLHS()));
    Expr* RHS = cast<Expr>(Visit(BinOp->getRHS()));
    BinaryOperatorKind opCode = BinOp->getOpcode();

    if (opCode == BO_Mul) {
      // 0 * smth or smth * 0 == 0
       if (evalsToZero(LHS, m_Context))
         return LHS;
       if (evalsToZero(RHS, m_Context))
         return RHS;

       // 1 * smth or smth * 1 == smth
       if (evalsToOne(LHS, m_Context))
         return RHS;
       if (evalsToOne(RHS, m_Context))
         return LHS;
    }
    else if (opCode == BO_Add || opCode == BO_Sub) {
      // smth +- 0 == smth
      if (evalsToZero(RHS, m_Context))
        return LHS;

      // 0 + smth == smth
      if (opCode == BO_Add)
        if (evalsToZero(LHS, m_Context))
          return RHS;
    }
    else if (opCode == BO_Div) {
      // 0 / smth == 0
      if (evalsToZero(LHS, m_Context))
        return LHS;
    }

    BinOp->setLHS(trivialFold(LHS));
    BinOp->setRHS(trivialFold(RHS));
    return BinOp;
  }

  Expr* ConstantFolder::VisitParenExpr(clang::ParenExpr* PE) {
    Expr* result = cast<Expr>(Visit(PE->getSubExpr()));
    if (!isa<BinaryOperator>(result))
      return result;
    PE->setSubExpr(result);
    return PE;
  }

  Expr* ConstantFolder::fold(Expr* E) {
    if (!m_Enabled)
      return E;

    Expr* result = Visit(E);

    return cast<Expr>(result);
  }

  Expr* ConstantFolder::synthesizeLiteral(QualType QT, ASTContext& C,
                                          uint64_t val) {
    // SourceLocation noLoc;
    Expr* Result = 0;
    QT = QT.getCanonicalType();
    if (QT->isEnumeralType()) {
      llvm::APInt APVal(C.getIntWidth(QT), val,
                        QT->isSignedIntegerOrEnumerationType());
      Result = clad::synthesizeLiteral(
          dyn_cast<EnumType>(QT)->getDecl()->getIntegerType(), C, APVal);
      SourceLocation noLoc;
      Expr* cast = CXXStaticCastExpr::Create(
          C, QT, CLAD_COMPAT_ExprValueKind_R_or_PR_Value,
          clang::CastKind::CK_IntegralCast, Result, /*CXXCastPath=*/nullptr,
          C.getTrivialTypeSourceInfo(QT, noLoc)
              CLAD_COMPAT_CLANG12_CastExpr_DefaultFPO,
          noLoc, noLoc, SourceRange());
      Result = cast;
    } else if (QT->isPointerType()) {
      Result = clad::synthesizeLiteral(QT, C);
    } else if (QT->isBooleanType()) {
      Result = clad::synthesizeLiteral(QT, C, (bool)val);
    } else if (QT->isIntegralType(C)) {
      if (QT->isAnyCharacterType())
        QT = C.IntTy;
      if (const auto* BT = dyn_cast<BuiltinType>(QT.getTypePtr()))
        if (BT->getKind() == BuiltinType::Short)
          QT = C.IntTy;
      llvm::APInt APVal(C.getIntWidth(QT), val,
                         QT->isSignedIntegerOrEnumerationType());
      Result = clad::synthesizeLiteral(QT, C, APVal);
    } else if (QT->isRealFloatingType()) {
      llvm::APFloat APVal(C.getFloatTypeSemantics(QT), val);
      Result = clad::synthesizeLiteral(QT, C, APVal);
    } else {
      // FIXME: Handle other types, like Complex, Structs, typedefs, etc.
      // typecasting may be needed right now
      Result = ConstantFolder::synthesizeLiteral(C.IntTy, C, val);
    }
    assert(Result && "Unsupported type for constant folding.");
    return Result;
  }
} // end namespace clad
