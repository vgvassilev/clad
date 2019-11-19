//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// author:  Alexander Penev <alexander_penev@yahoo.com>
//------------------------------------------------------------------------------

#ifndef CLAD_COMPATIBILITY
#define CLAD_COMPATIBILITY

#include "clang/Basic/Version.h"
#include "llvm/Config/llvm-config.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/Sema/Sema.h"

namespace clad_compat {

using namespace clang;
using namespace llvm;

// Compatibility helper function for creation CompoundStmt. Clang 6 and above use Create.

static inline CompoundStmt* CompoundStmt_Create(
        const ASTContext &Ctx, ArrayRef<Stmt *> Stmts,
        SourceLocation LB, SourceLocation RB)
{
#if CLANG_VERSION_MAJOR == 5
   return new (Ctx) CompoundStmt(Ctx, Stmts, LB, RB);
#elif CLANG_VERSION_MAJOR >= 6
   return CompoundStmt::Create(Ctx, Stmts, LB, RB);
#endif
}


// Clang 6 rename Sema::ForRedeclaration to Sema::ForVisibleRedeclaration

#if CLANG_VERSION_MAJOR == 5
   const auto Sema_ForVisibleRedeclaration = Sema::ForRedeclaration;
#elif CLANG_VERSION_MAJOR >= 6
   const auto Sema_ForVisibleRedeclaration = Sema::ForVisibleRedeclaration;
#endif


// Clang 6 rename Declarator to DeclaratorContext, but Declarator is used
// as name for another class.

#if CLANG_VERSION_MAJOR == 5
   using DeclaratorContext = Declarator;
#elif CLANG_VERSION_MAJOR >= 6
   using DeclaratorContext = DeclaratorContext;
#endif


// Clang 7 add one extra param in UnaryOperator constructor.

#if CLANG_VERSION_MAJOR < 7
   #define CLAD_COMPAT_CLANG7_UnaryOperator_ExtraParams /**/
#elif CLANG_VERSION_MAJOR >= 7
   #define CLAD_COMPAT_CLANG7_UnaryOperator_ExtraParams ,Node->canOverflow()
#endif


// Clang 8 change E->EvaluateAsInt(APSInt int, context) ===> E->EvaluateAsInt(Expr::EvalResult res, context)

static bool Expr_EvaluateAsInt(const Expr *E,
                         APSInt &IntValue, const ASTContext &Ctx,
                         Expr::SideEffectsKind AllowSideEffects = Expr::SideEffectsKind::SE_NoSideEffects)
{
#if CLANG_VERSION_MAJOR < 8
   return E->EvaluateAsInt(IntValue, Ctx, AllowSideEffects);
#elif CLANG_VERSION_MAJOR >= 8
   Expr::EvalResult res;
   if (E->EvaluateAsInt(res, Ctx, AllowSideEffects)) {
     IntValue = res.Val.getInt();
     return true;
   }
   return false;
#endif
}

// Compatibility helper function for creation IfStmt. Clang 8 and above use Create.

static inline IfStmt* IfStmt_Create(const ASTContext &Ctx,
   SourceLocation IL, bool IsConstexpr,
   Stmt *Init, VarDecl *Var, Expr *Cond,
   Stmt *Then, SourceLocation EL=SourceLocation(), Stmt *Else=nullptr)
{
#if CLANG_VERSION_MAJOR < 8
   return new (Ctx) IfStmt(Ctx, IL, IsConstexpr, Init, Var, Cond, Then, EL, Else);
#elif CLANG_VERSION_MAJOR >= 8
   return IfStmt::Create(Ctx, IL, IsConstexpr, Init, Var, Cond, Then, EL, Else);
#endif
}


// Clang 8 change Node->getIdentType() ===> Node->getIdentKind()

#if CLANG_VERSION_MAJOR < 8
   #define getIdentKind() getIdentType()
#endif


// Clang 8 change <NAME>Stmt(...) constructor to private ===> Use <NAME>Stmt::Create(...)

#if CLANG_VERSION_MAJOR < 8
   #define CLAD_COMPAT_CREATE(CLASS, CTORARGS) (new (Ctx) CLASS CTORARGS)
#elif CLANG_VERSION_MAJOR >= 8
   #define CLAD_COMPAT_CREATE(CLASS, CTORARGS) (CLASS::Create CTORARGS)
#endif


// Compatibility helper function for creation CallExpr. Clang 8 and above use Create.

static inline CallExpr* CallExpr_Create(const ASTContext &Ctx, Expr *Fn, ArrayRef< Expr *> Args,
   QualType Ty, ExprValueKind VK, SourceLocation RParenLoc)
{
#if CLANG_VERSION_MAJOR < 8
   return new (Ctx) CallExpr(Ctx, Fn, Args, Ty, VK, RParenLoc);
#elif CLANG_VERSION_MAJOR >= 8
   return CallExpr::Create(Ctx, Fn, Args, Ty, VK, RParenLoc);
#endif
}


// Compatibility helper function for creation CXXOperatorCallExpr. Clang 8 and above use Create.

static inline CXXOperatorCallExpr* CXXOperatorCallExpr_Create(ASTContext &Ctx,
   OverloadedOperatorKind OpKind, Expr *Fn, ArrayRef<Expr *> Args, QualType Ty,
   ExprValueKind VK, SourceLocation OperatorLoc, FPOptions FPFeatures)
{
#if CLANG_VERSION_MAJOR < 8
   return new (Ctx) CXXOperatorCallExpr(Ctx, OpKind, Fn, Args, Ty, VK, OperatorLoc, FPFeatures);
#elif CLANG_VERSION_MAJOR >= 8
   return CXXOperatorCallExpr::Create(Ctx, OpKind, Fn, Args, Ty, VK, OperatorLoc, FPFeatures);
#endif
}


// Compatibility helper function for creation CXXMemberCallExpr. Clang 8 and above use Create.

static inline CXXMemberCallExpr* CXXMemberCallExpr_Create(ASTContext &Ctx,
   Expr *Fn, ArrayRef<Expr *> Args, QualType Ty, ExprValueKind VK, SourceLocation RP)
{
#if CLANG_VERSION_MAJOR < 8
   return new (Ctx) CXXMemberCallExpr(Ctx, Fn, Args, Ty, VK, RP);
#elif CLANG_VERSION_MAJOR >= 8
   return CXXMemberCallExpr::Create(const_cast<ASTContext&>(Ctx), Fn, Args, Ty, VK, RP);
#endif
}


// Compatibility helper function for creation CaseStmt. Clang 8 and above use Create.

static inline CaseStmt* CaseStmt_Create(ASTContext &Ctx,
   Expr *lhs, Expr *rhs, SourceLocation caseLoc, SourceLocation ellipsisLoc, SourceLocation colonLoc)
{
#if CLANG_VERSION_MAJOR < 8
   return new (Ctx) CaseStmt(lhs, rhs, caseLoc, ellipsisLoc, colonLoc);
#elif CLANG_VERSION_MAJOR >= 8
   return CaseStmt::Create(const_cast<ASTContext&>(Ctx), lhs, rhs, caseLoc, ellipsisLoc, colonLoc);
#endif
}


// Compatibility helper function for creation SwitchStmt. Clang 8 and above use Create.

static inline SwitchStmt* SwitchStmt_Create(const ASTContext &Ctx,
   Stmt *Init, VarDecl *Var, Expr *Cond)
{
#if CLANG_VERSION_MAJOR < 8
   return new (Ctx) SwitchStmt(Ctx, Init, Var, Cond);
#elif CLANG_VERSION_MAJOR >= 8
   return SwitchStmt::Create(Ctx, Init, Var, Cond);
#endif
}


// Clang 8 change E->getLocStart() ===> E->getBeginLoc()
// E->getLocEnd() ===> E->getEndLoc()

#if CLANG_VERSION_MAJOR < 8
   #define getBeginLoc() getLocStart()
   #define getEndLoc() getLocEnd()
#endif


// Clang 8 add one extra param (Ctx) in some constructors.

#if CLANG_VERSION_MAJOR < 8
   #define CLAD_COMPAT_CLANG8_Ctx_ExtraParams /**/
#elif CLANG_VERSION_MAJOR >= 7
   #define CLAD_COMPAT_CLANG8_Ctx_ExtraParams Ctx,
#endif


// Clang 8 change result->setNumArgs(Ctx, Num) ===> result->setNumArgsUnsafe(Num)

#if CLANG_VERSION_MAJOR < 8
   #define setNumArgsUnsafe(NUM) setNumArgs(Ctx, NUM)
#endif


} // namespace clad_compat

#endif //CLAD_COMPATIBILITY
