//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// author:  Alexander Penev <alexander_penev@yahoo.com>
//------------------------------------------------------------------------------

#ifndef CLAD_COMPATIBILITY
#define CLAD_COMPATIBILITY

#include "llvm/Config/llvm-config.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/Version.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Sema/Sema.h"

namespace clad_compat {

using namespace clang;
using namespace llvm;

// Compatibility helper function for creation CompoundStmt. Clang 6 and above use Create.

static inline bool SourceManager_isPointWithin(const SourceManager& SM,
                                               SourceLocation Loc,
                                               SourceLocation B,
                                               SourceLocation E) {
#if CLANG_VERSION_MAJOR == 5
  return Loc == B || Loc == E || (SM.isBeforeInTranslationUnit(B, Loc) &&
                                  SM.isBeforeInTranslationUnit(Loc, E));
#elif CLANG_VERSION_MAJOR >= 6
  return SM.isPointWithin(Loc, B, E);
#endif
}

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

static inline bool Expr_EvaluateAsInt(const Expr *E,
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

// Compatibility helper function for creation IfStmt.
// Clang 8 and above use Create.
// Clang 12 and above use two extra params.

static inline IfStmt* IfStmt_Create(const ASTContext &Ctx,
   SourceLocation IL, bool IsConstexpr,
   Stmt *Init, VarDecl *Var, Expr *Cond,
   SourceLocation LPL, SourceLocation RPL,
   Stmt *Then, SourceLocation EL=SourceLocation(), Stmt *Else=nullptr)
{
#if CLANG_VERSION_MAJOR < 8
   return new (Ctx) IfStmt(Ctx, IL, IsConstexpr, Init, Var, Cond, Then, EL, Else);
#elif CLANG_VERSION_MAJOR < 12
   return IfStmt::Create(Ctx, IL, IsConstexpr, Init, Var, Cond, Then, EL, Else);
#elif CLANG_VERSION_MAJOR >= 12
   return IfStmt::Create(Ctx, IL, IsConstexpr, Init, Var, Cond, LPL, RPL, Then, EL, Else);
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


// Compatibility helper function for creation CallExpr.
// Clang 8 and above use Create.
// Clang 12 and above use one extra param.

#if CLANG_VERSION_MAJOR < 8
static inline CallExpr* CallExpr_Create(const ASTContext &Ctx, Expr *Fn, ArrayRef< Expr *> Args,
   QualType Ty, ExprValueKind VK, SourceLocation RParenLoc)
{
   return new (Ctx) CallExpr(Ctx, Fn, Args, Ty, VK, RParenLoc);
}
#elif CLANG_VERSION_MAJOR < 12
static inline CallExpr* CallExpr_Create(const ASTContext &Ctx, Expr *Fn, ArrayRef< Expr *> Args,
   QualType Ty, ExprValueKind VK, SourceLocation RParenLoc,
   unsigned MinNumArgs = 0, CallExpr::ADLCallKind UsesADL = CallExpr::NotADL)
{
   return CallExpr::Create(Ctx, Fn, Args, Ty, VK, RParenLoc, MinNumArgs, UsesADL);
}
#elif CLANG_VERSION_MAJOR >= 12
static inline CallExpr* CallExpr_Create(const ASTContext &Ctx, Expr *Fn, ArrayRef< Expr *> Args,
   QualType Ty, ExprValueKind VK, SourceLocation RParenLoc, FPOptionsOverride FPFeatures,
   unsigned MinNumArgs = 0, CallExpr::ADLCallKind UsesADL = CallExpr::NotADL)
{
   return CallExpr::Create(Ctx, Fn, Args, Ty, VK, RParenLoc, FPFeatures, MinNumArgs, UsesADL);
}
#endif


// Clang 8 add one extra param (Ctx) in some constructors.
// Clang 12 and above use one extra param.

#if CLANG_VERSION_MAJOR < 8
   #define CLAD_COMPAT_CLANG8_CallExpr_ExtraParams /**/
#elif CLANG_VERSION_MAJOR < 12
   #define CLAD_COMPAT_CLANG8_CallExpr_ExtraParams ,Node->getNumArgs(),Node->getADLCallKind()
#elif CLANG_VERSION_MAJOR >= 12
   #define CLAD_COMPAT_CLANG8_CallExpr_ExtraParams ,Node->getFPFeatures(),Node->getNumArgs(),Node->getADLCallKind()
#endif

// Clang 11 add one param to CXXOperatorCallExpr_Create, ADLCallKind, and many other constructor and Create changes

#if CLANG_VERSION_MAJOR < 11

static inline void ExprSetDeps(Expr* result, Expr* Node) {
   result->setValueDependent(Node->isValueDependent());
   result->setTypeDependent(Node->isTypeDependent());
}

   #define CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParams /**/
   #define CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsPar /**/
   #define CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsUse /**/
   #define CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsOverride FPOptions
   #define CLAD_COMPAT_CLANG11_LangOptions_EtraParams /**/
   #define CLAD_COMPAT_CLANG11_Ctx_ExtraParams /**/
   #define CLAD_COMPAT_CREATE11(CLASS, CTORARGS) (new (Ctx) CLASS CTORARGS)
   #define CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Removed Node->getComputationLHSType(),Node->getComputationResultType(),
   #define CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Moved /**/
   #define CLAD_COMPAT_CLANG11_ChooseExpr_EtraParams_Removed ,Node->isTypeDependent(),Node->isValueDependent()
   #define CLAD_COMPAT_CLANG11_WhileStmt_ExtraParams /**/

#elif CLANG_VERSION_MAJOR >= 11

struct ExprDependenceAccessor : public Expr {
   void setDependence(ExprDependence Deps) {
      Expr::setDependence(Deps);
   }
};

static inline void ExprSetDeps(Expr* result, Expr* Node) {
   ((ExprDependenceAccessor*)result)->setDependence(Node->getDependence());
}

   #define CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParams ,Node->getADLCallKind()
   #define CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsPar ,clang::CallExpr::ADLCallKind UsesADL
   #define CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsUse ,UsesADL
   #define CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsOverride FPOptionsOverride
   #define CLAD_COMPAT_CLANG11_LangOptions_EtraParams Ctx.getLangOpts()
   #define CLAD_COMPAT_CLANG11_Ctx_ExtraParams Ctx,
   #define CLAD_COMPAT_CREATE11(CLASS, CTORARGS) (CLASS::Create CTORARGS)
   #define CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Removed /**/
   #define CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Moved ,Node->getComputationLHSType(),Node->getComputationResultType()
   #define CLAD_COMPAT_CLANG11_ChooseExpr_EtraParams_Removed /**/
   #define CLAD_COMPAT_CLANG11_WhileStmt_ExtraParams ,Node->getLParenLoc(),Node->getRParenLoc()
#endif

// Compatibility helper function for creation CXXOperatorCallExpr. Clang 8 and above use Create.

static inline CXXOperatorCallExpr* CXXOperatorCallExpr_Create(ASTContext &Ctx,
   OverloadedOperatorKind OpKind, Expr *Fn, ArrayRef<Expr *> Args, QualType Ty,
   ExprValueKind VK, SourceLocation OperatorLoc, CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsOverride FPFeatures
   CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsPar
   )
{
#if CLANG_VERSION_MAJOR < 8
   return new (Ctx) CXXOperatorCallExpr(Ctx, OpKind, Fn, Args, Ty, VK, OperatorLoc, FPFeatures);
#elif CLANG_VERSION_MAJOR >= 8
   return CXXOperatorCallExpr::Create(Ctx, OpKind, Fn, Args, Ty, VK, OperatorLoc, FPFeatures
   CLAD_COMPAT_CLANG11_CXXOperatorCallExpr_Create_ExtraParamsUse
   );
#endif
}


// Compatibility helper function for creation CXXMemberCallExpr.
// Clang 8 and above use Create.
// Clang 12 and above use two extra param.

#if CLANG_VERSION_MAJOR < 8
static inline CXXMemberCallExpr* CXXMemberCallExpr_Create(ASTContext &Ctx,
   Expr *Fn, ArrayRef<Expr *> Args, QualType Ty, ExprValueKind VK, SourceLocation RP)
{
   return new (Ctx) CXXMemberCallExpr(Ctx, Fn, Args, Ty, VK, RP);
}
#elif CLANG_VERSION_MAJOR < 12
static inline CXXMemberCallExpr* CXXMemberCallExpr_Create(ASTContext &Ctx,
   Expr *Fn, ArrayRef<Expr *> Args, QualType Ty, ExprValueKind VK, SourceLocation RP)
{
   return CXXMemberCallExpr::Create(const_cast<ASTContext&>(Ctx), Fn, Args, Ty, VK, RP);
}
#elif CLANG_VERSION_MAJOR >= 12
static inline CXXMemberCallExpr* CXXMemberCallExpr_Create(ASTContext &Ctx,
   Expr *Fn, ArrayRef<Expr *> Args, QualType Ty, ExprValueKind VK, SourceLocation RP,
   FPOptionsOverride FPFeatures, unsigned MinNumArgs = 0)
{
   return CXXMemberCallExpr::Create(const_cast<ASTContext&>(Ctx), Fn, Args, Ty, VK, RP,
                                    FPFeatures, MinNumArgs);
}
#endif


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


// Compatibility helper function for creation SwitchStmt.
// Clang 8 and above use Create.
// Clang 12 and above use two extra params.

static inline SwitchStmt* SwitchStmt_Create(const ASTContext &Ctx,
   Stmt *Init, VarDecl *Var, Expr *Cond,
   SourceLocation LParenLoc, SourceLocation RParenLoc)
{
#if CLANG_VERSION_MAJOR < 8
   return new (Ctx) SwitchStmt(Ctx, Init, Var, Cond);
#elif CLANG_VERSION_MAJOR < 12
   return SwitchStmt::Create(Ctx, Init, Var, Cond);
#elif CLANG_VERSION_MAJOR >= 12
   return SwitchStmt::Create(Ctx, Init, Var, Cond, LParenLoc, RParenLoc);
#endif
}


// Clang 8 change E->getLocStart() ===> E->getBeginLoc()
// E->getLocEnd() ===> E->getEndLoc()
// Clang 7 define both for compatibility

#if CLANG_VERSION_MAJOR < 7
   #define getBeginLoc() getLocStart()
   #define getEndLoc() getLocEnd()
#endif


// Clang 8 add one extra param (Ctx) in some constructors.

#if CLANG_VERSION_MAJOR < 8
   #define CLAD_COMPAT_CLANG8_Ctx_ExtraParams /**/
#elif CLANG_VERSION_MAJOR >= 8
   #define CLAD_COMPAT_CLANG8_Ctx_ExtraParams Ctx,
#endif


// Clang 8 change result->setNumArgs(Ctx, Num) ===> result->setNumArgsUnsafe(Num)

#if CLANG_VERSION_MAJOR < 8
   #define setNumArgsUnsafe(NUM) setNumArgs(Ctx, NUM)
#endif


// Compatibility helper function for getConstexprKind(). Clang 9

template<class T>
static inline T GetResult(ActionResult<T> Res)
{
   return Res.get();
}


// Compatibility helper function for getConstexprKind(). Clang 9 define new method
// ConstexprKind getConstexprKing() and old bool isConstexpr().

#if CLANG_VERSION_MAJOR < 9
static inline bool Function_GetConstexprKind(const FunctionDecl* F)
{
   return F->isConstexpr();
}
#elif CLANG_VERSION_MAJOR >= 9
static inline ConstexprSpecKind Function_GetConstexprKind(const FunctionDecl* F)
{
   return F->getConstexprKind();
}
#endif


// Clang 9 add one extra param (Ctx) in some constructors.

#if CLANG_VERSION_MAJOR < 9
   #define CLAD_COMPAT_CLANG9_MemberExpr_ExtraParams(NOUR) /**/
#elif CLANG_VERSION_MAJOR >= 9
   #define CLAD_COMPAT_CLANG9_MemberExpr_ExtraParams(NOUR) ,NOUR
#endif



// Clang 9 change PragmaIntroducerKind ===> PragmaIntroducer.

#if CLANG_VERSION_MAJOR < 9
   #define PragmaIntroducer PragmaIntroducerKind
#endif


// Clang 10 change add new param in getConstantArrayType.

static inline QualType getConstantArrayType(const ASTContext &Ctx,
   QualType EltTy,
   const APInt &ArySize,
   const Expr* SizeExpr,
   clang::ArrayType::ArraySizeModifier ASM,
   unsigned IndexTypeQuals)
{
#if CLANG_VERSION_MAJOR < 10
   return Ctx.getConstantArrayType(EltTy, ArySize, ASM, IndexTypeQuals);
#elif CLANG_VERSION_MAJOR >= 10
   return Ctx.getConstantArrayType(EltTy, ArySize, SizeExpr, ASM, IndexTypeQuals);
#endif
}

// Clang 10 add new last param TrailingRequiresClause in FunctionDecl::Create

#if CLANG_VERSION_MAJOR < 10
   #define CLAD_COMPAT_CLANG10_FunctionDecl_Create_ExtraParams(x) /**/
#elif CLANG_VERSION_MAJOR >= 10
   #define CLAD_COMPAT_CLANG10_FunctionDecl_Create_ExtraParams(x) ,((x)?VD.Clone((x)):nullptr)
#endif

// Clang 10 remove GetTemporaryExpr(). Use getSubExpr() instead

#if CLANG_VERSION_MAJOR < 10
   #define CLAD_COMPAT_CLANG10_GetTemporaryExpr(x) (x)->GetTemporaryExpr()
#elif CLANG_VERSION_MAJOR >= 10
   #define CLAD_COMPAT_CLANG10_GetTemporaryExpr(x) (x)->getSubExpr()?Clone((x)->getSubExpr()):nullptr
#endif

// Clang 10 add one param to StmtExpr constructor: unsigned TemplateDepth

#if CLANG_VERSION_MAJOR < 10
   #define CLAD_COMPAT_CLANG10_StmtExpr_Create_ExtraParams /**/
#elif CLANG_VERSION_MAJOR >= 10
   #define CLAD_COMPAT_CLANG10_StmtExpr_Create_ExtraParams ,Node->getTemplateDepth()
#endif

// Clang 11 add one extra param in UnaryOperator constructor.

#if CLANG_VERSION_MAJOR < 11
   #define CLAD_COMPAT_CLANG11_UnaryOperator_ExtraParams /**/
#elif CLANG_VERSION_MAJOR >= 11
   #define CLAD_COMPAT_CLANG11_UnaryOperator_ExtraParams ,Node->getFPOptionsOverride()
#endif


// Clang 12 rename DeclaratorContext::LambdaExprContext to DeclaratorContext::LambdaExpr.

#if CLANG_VERSION_MAJOR < 12
   #define CLAD_COMPAT_CLANG12_Declarator_LambdaExpr LambdaExprContext
#elif CLANG_VERSION_MAJOR >= 12
   #define CLAD_COMPAT_CLANG12_Declarator_LambdaExpr LambdaExpr
#endif

// Clang 12 add one extra param (FPO) that we get from Node in Create method of:
// ImplicitCastExpr, CStyleCastExpr, CXXStaticCastExpr and CXXFunctionalCastExpr

#if CLANG_VERSION_MAJOR < 12
   #define CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node) /**/
#elif CLANG_VERSION_MAJOR >= 12
   #define CLAD_COMPAT_CLANG12_CastExpr_GetFPO(Node) ,Node->getFPFeatures()
#endif

// Clang 12 adds one extra param (FPO) in Create method of:
// ImplicitCastExpr, CStyleCastExpr, CXXStaticCastExpr and CXXFunctionalCastExpr

#if CLANG_VERSION_MAJOR < 12
#define CLAD_COMPAT_CLANG12_CastExpr_DefaultFPO /**/
#elif CLANG_VERSION_MAJOR >= 12
#define CLAD_COMPAT_CLANG12_CastExpr_DefaultFPO , FPOptionsOverride()
#endif

// Clang 12 add two extra param (Left and Right paren location) in Create method of:
// IfStat::Create

#if CLANG_VERSION_MAJOR < 12
   #define CLAD_COMPAT_CLANG12_LR_ExtraParams(Node) /**/
#elif CLANG_VERSION_MAJOR >= 12
   #define CLAD_COMPAT_CLANG12_LR_ExtraParams(Node) ,Node->getLParenLoc(),Node->getRParenLoc()
#endif

/// In Clang < 8, `CXXMethodDecl::getThisType()` member function requires
/// `ASTContext` to be passed as an argument.
static inline QualType CXXMethodDecl_getThisType(Sema& SemaRef,
                                                 const CXXMethodDecl* method) {
#if CLANG_VERSION_MAJOR >= 8
  auto thisType = method->getThisType();
#elif CLANG_VERSION_MAJOR < 8
  auto thisType = method->getThisType(SemaRef.getASTContext());
#endif
  return thisType;
}

/// Clang < 9, do not provide `Sema::BuildCXXThisExpr` function.
static inline CXXThisExpr* Sema_BuildCXXThisExpr(Sema& SemaRef,
                                                 const CXXMethodDecl* method) {
  auto thisType = CXXMethodDecl_getThisType(SemaRef, method);
  SourceLocation noLoc;
#if CLANG_VERSION_MAJOR >= 9
  return cast<CXXThisExpr>(
      SemaRef.BuildCXXThisExpr(noLoc, thisType, /*IsImplicit=*/true));
#elif CLANG_VERSION_MAJOR < 9
  auto thisExpr = new (SemaRef.getASTContext())
      CXXThisExpr(noLoc, thisType, /*IsImplicit=*/true);
  SemaRef.CheckCXXThisCapture(thisExpr->getExprLoc());
  return thisExpr;
#endif
}

/// clang >= 11 added more source locations parameters in `Sema::ActOnWhileStmt`
static inline StmtResult
Sema_ActOnWhileStmt(Sema& SemaRef, Sema::ConditionResult cond, Stmt* body) {
  SourceLocation noLoc;
#if CLANG_VERSION_MAJOR < 11
  return SemaRef.ActOnWhileStmt(/*WhileLoc=*/noLoc, cond, body);
#elif CLANG_VERSION_MAJOR >= 11
  return SemaRef.ActOnWhileStmt(/*WhileLoc=*/noLoc, /*LParenLoc=*/noLoc, cond,
                                /*RParenLoc=*/noLoc, body);
#endif
}

/// Clang >= 12 has more source locations parameters in `Sema::ActOnStartOfSwitchStmt`
static inline StmtResult
Sema_ActOnStartOfSwitchStmt(Sema& SemaRef, Stmt* initStmt,
                            Sema::ConditionResult Cond) {
  SourceLocation noLoc;
#if CLANG_VERSION_MAJOR >= 12
  return SemaRef.ActOnStartOfSwitchStmt(
      /*SwitchLoc=*/noLoc,
      /*LParenLoc=*/noLoc, initStmt, Cond,
      /*RParenLoc=*/noLoc);
#elif CLANG_VERSION_MAJOR < 12
  return SemaRef.ActOnStartOfSwitchStmt(/*SwitchLoc=*/noLoc, initStmt, Cond);
#endif
}

/// Clang 9 added an extra parameter for result storage kind in
/// ConstantExpr::Create 
/// Clang 11 added an extra parameter for immediate invocation in
/// ConstantExpr::Create
#if CLANG_VERSION_MAJOR < 9
#define CLAD_COMPAT_ConstantExpr_Create_ExtraParams
#elif CLANG_VERSION_MAJOR < 11
#define CLAD_COMPAT_ConstantExpr_Create_ExtraParams\
  , Node->getResultStorageKind()
#elif CLANG_VERSION_MAJOR >= 11
#define CLAD_COMPAT_ConstantExpr_Create_ExtraParams\
  , Node->getResultStorageKind(), Node->isImmediateInvocation()
#endif

#if CLANG_VERSION_MAJOR < 13
#define CLAD_COMPAT_ExprValueKind_R_or_PR_Value ExprValueKind::VK_RValue
#elif CLANG_VERSION_MAJOR >= 13
#define CLAD_COMPAT_ExprValueKind_R_or_PR_Value ExprValueKind::VK_PRValue
#endif

#if LLVM_VERSION_MAJOR < 13
#define CLAD_COMPAT_llvm_sys_fs_Append llvm::sys::fs::F_Append
#elif LLVM_VERSION_MAJOR >= 13
#define CLAD_COMPAT_llvm_sys_fs_Append llvm::sys::fs::OF_Append
#endif
} // namespace clad_compat

#endif //CLAD_COMPATIBILITY
