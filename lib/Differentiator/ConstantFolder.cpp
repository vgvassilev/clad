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
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace clad {
  using namespace clang;

  // `sizeof`, `alignof` and `offsetof` are constants, but their value belongs
  // to the target ABI, while the folder evaluates on the host. Folding one
  // into a numeric literal -- or folding an identity keyed on its host value
  // -- bakes the host's answer into generated code that may be compiled for a
  // different target (a CUDA derivative printing `n * 8UL` instead of
  // `n * sizeof(double)`). Treat any expression containing one as opaque.
  static bool containsTargetDependentConstant(const Expr* E) {
    llvm::SmallVector<const Stmt*, 8> pending{E};
    while (!pending.empty()) {
      const Stmt* S = pending.pop_back_val();
      // Null child slots (a ForStmt's empty init, ...) only occur on
      // statements, which enter an expression tree via a GNU StmtExpr --
      // and clad hoists those into temporaries before folding sees them.
      if (!S)
        continue; // LCOV_EXCL_LINE: defensive, see above
      if (isa<UnaryExprOrTypeTraitExpr>(S) || isa<OffsetOfExpr>(S))
        return true;
      pending.append(S->child_begin(), S->child_end());
    }
    return false;
  }

  // The one evaluation gate for every folding decision. EvaluateAsRValue
  // folds through side effects and only records them in HasSideEffects;
  // without that check an operand like `(counter++, 0.)` reads as a plain
  // zero and dropping or literalizing it deletes the increment from the
  // derivative.
  static bool evaluatesForFolding(Expr* E, ASTContext& C,
                                  Expr::EvalResult& Result) {
    return E->EvaluateAsRValue(Result, C) && !Result.HasSideEffects &&
           !containsTargetDependentConstant(E);
  }

  static bool evalsToN(Expr* E, ASTContext& C, uint64_t N = 0) {
    Expr::EvalResult result;
    if (evaluatesForFolding(E, C, result)) {
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

  static bool evalsToOne(Expr* E, ASTContext& C) {
    return evalsToN(E, C, /*N=*/1);
  }

  static Expr* synthesizeLiteral(QualType QT, ASTContext& C,
                                 const llvm::APInt& val) {
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
    // Already a literal (possibly behind an implicit conversion): synthesizing
    // a fresh one would only re-spell it at the converted type (`x *= 2`
    // becoming `x *= 2.`), churning generated code without simplifying it.
    const Expr* Inner = E->IgnoreImpCasts();
    if (isa<IntegerLiteral>(Inner) || isa<FloatingLiteral>(Inner) ||
        isa<CharacterLiteral>(Inner) || isa<CXXBoolLiteralExpr>(Inner) ||
        isa<CXXNullPtrLiteralExpr>(Inner))
      return E;
    Expr::EvalResult Result;
    if (evaluatesForFolding(E, m_Context, Result)) {
      if (Result.Val.isFloat()) {
        llvm::APFloat F = Result.Val.getFloat();
        E = clad::synthesizeLiteral(E->getType(), m_Context, F);
      }
      else if (Result.Val.isInt()) {
        llvm::APSInt I = Result.Val.getInt();
        QualType QT = E->getType();
        // A bool result needs a bool literal: an IntegerLiteral of boolean
        // type is malformed and crashes the statement printer when it looks
        // for an integer suffix.
        if (QT->isBooleanType())
          E = clad::synthesizeLiteral(QT, m_Context, I.getBoolValue());
        else if (QT->isIntegralType(m_Context))
          E = clad::synthesizeLiteral(QT, m_Context, I);
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
      if (evaluatesToZero(LHS, m_Context))
        return LHS;
      if (evaluatesToZero(RHS, m_Context))
        return RHS;

      // 1 * smth or smth * 1 == smth
      if (evalsToOne(LHS, m_Context))
        return RHS;
      if (evalsToOne(RHS, m_Context))
        return LHS;
    }
    else if (opCode == BO_Add || opCode == BO_Sub) {
      // smth +- 0 == smth
      if (evaluatesToZero(RHS, m_Context))
        return LHS;

      // 0 + smth == smth
      if (opCode == BO_Add)
        if (evaluatesToZero(LHS, m_Context))
          return RHS;
    }
    else if (opCode == BO_Div) {
      // 0 / smth == 0
      if (evaluatesToZero(LHS, m_Context))
        return LHS;
    }

    BinOp->setLHS(trivialFold(LHS));
    BinOp->setRHS(trivialFold(RHS));
    return BinOp;
  }

  /// True for expressions that never need parentheses around them: they bind
  /// tighter than any operator clad builds around a folded subexpression.
  static bool bindsTighterThanAnyOperator(const Expr* E) {
    E = E->IgnoreImpCasts();
    // An overloaded operator is a CallExpr in the AST but prints in operator
    // syntax, which binds like the operator it names; conservatively keep the
    // parentheses around it. Unreached today: every call that survives into a
    // derivative has been rebuilt by the visitors in function-call syntax.
    if (isa<CXXOperatorCallExpr>(E))
      return false; // LCOV_EXCL_LINE: defensive, see above
    return isa<DeclRefExpr>(E) || isa<IntegerLiteral>(E) ||
           isa<FloatingLiteral>(E) || isa<CXXBoolLiteralExpr>(E) ||
           isa<CXXNullPtrLiteralExpr>(E) || isa<MemberExpr>(E) ||
           isa<ArraySubscriptExpr>(E) || isa<CallExpr>(E) || isa<ParenExpr>(E);
  }

  Expr* ConstantFolder::VisitParenExpr(clang::ParenExpr* PE) {
    Expr* result = cast<Expr>(Visit(PE->getSubExpr()));
    // Clang's printer relies on ParenExpr for precedence, so dropping one
    // around anything that binds loosely makes -fdump-derived-fn print source
    // that no longer reparses to the AST it came from -- a parenthesized
    // conditional under a `*` prints as `a * c ? x : y`.
    if (!bindsTighterThanAnyOperator(result)) {
      PE->setSubExpr(result);
      return PE;
    }
    return result;
  }

  bool ConstantFolder::evaluatesToZero(Expr* E, ASTContext& C) {
    return E && evalsToN(E, C, /*N=*/0);
  }

  Expr* ConstantFolder::fold(Expr* E) {
    Expr* result = cast<Expr>(Visit(E));

    // The identity rules (`x + 0`, `1 * x`) hand back an operand, and an
    // operand can be an lvalue where the operator result was a prvalue. A
    // folded tangent ends up in argument position next to its primal, and clad
    // differentiates an lvalue argument w.r.t. the declaration it names while a
    // prvalue argument only gets a temporary adjoint -- so a fold that changes
    // the value category also changes which parameters the callee's pullback is
    // requested for, and the tangent stops matching its primal. Keep the
    // unfolded expression in that case. The check looks through the
    // lvalue-to-rvalue conversion: it is the declaration underneath that the
    // reverse pass keys on, not the cast. Because this hands the whole input
    // back, callers must only pass expressions that are safe to re-evaluate
    // in full -- see the term selection in
    // BaseForwardModeVisitor::VisitBinaryOperator.
    if (!E->IgnoreParenImpCasts()->isLValue() &&
        result->IgnoreParenImpCasts()->isLValue())
      return E;

    return result;
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
          C.getTrivialTypeSourceInfo(QT, noLoc), FPOptionsOverride(), noLoc,
          noLoc, SourceRange());
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
