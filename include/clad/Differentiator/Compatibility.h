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

// clang-18 CXXThisExpr got extra argument

#if CLANG_VERSION_MAJOR > 17
#define CLAD_COMPAT_CLANG17_CXXThisExpr_ExtraParam DEFINE_CREATE_EXPR(CXXThisExpr, (Ctx,
#else
#define CLAD_COMPAT_CLANG17_CXXThisExpr_ExtraParam DEFINE_CLONE_EXPR(CXXThisExpr, (
#endif

// clang-18 ActOnLambdaExpr got extra argument

#if CLANG_VERSION_MAJOR > 17
#define CLAD_COMPAT_CLANG17_ActOnLambdaExpr_getCurrentScope_ExtraParam(V) /**/
#else
#define CLAD_COMPAT_CLANG17_ActOnLambdaExpr_getCurrentScope_ExtraParam(V)      \
  , (V).getCurrentScope()
#endif

// Clang 18 ArrayType::Normal -> ArraySizeModifier::Normal

#if LLVM_VERSION_MAJOR < 18
const auto ArraySizeModifier_Normal = clang::ArrayType::Normal;
#else
const auto ArraySizeModifier_Normal = clang::ArraySizeModifier::Normal;
#endif

// Compatibility helper function for creation UnresolvedLookupExpr.
// Clang-18 extra argument knowndependent.
// FIXME: Knowndependent set to false temporarily until known value found for
// initialisation.

static inline Stmt* UnresolvedLookupExpr_Create(
    const ASTContext& Ctx, CXXRecordDecl* NamingClass,
    NestedNameSpecifierLoc QualifierLoc, SourceLocation TemplateKWLoc,
    const DeclarationNameInfo& NameInfo, bool RequiresADL,
    const TemplateArgumentListInfo* Args, UnresolvedSetIterator Begin,
    UnresolvedSetIterator End) {

#if CLANG_VERSION_MAJOR < 18
  return UnresolvedLookupExpr::Create(Ctx, NamingClass, QualifierLoc,
                                      TemplateKWLoc, NameInfo, RequiresADL,
                                      // They get copied again by
                                      // OverloadExpr, so we are safe.
                                      Args, Begin, End);

#else
  bool KnownDependent = false;
  return UnresolvedLookupExpr::Create(Ctx, NamingClass, QualifierLoc,
                                      TemplateKWLoc, NameInfo, RequiresADL,
                                      // They get copied again by
                                      // OverloadExpr, so we are safe.
                                      Args, Begin, End, KnownDependent);
#endif
}

// Clang 18 ETK_None -> ElaboratedTypeKeyword::None

#if LLVM_VERSION_MAJOR < 18
const auto ElaboratedTypeKeyword_None = ETK_None;
#else
const auto ElaboratedTypeKeyword_None = ElaboratedTypeKeyword::None;
#endif

// Clang 18 endswith->ends_with
// and starstwith->starts_with

#if LLVM_VERSION_MAJOR < 18
#define starts_with startswith
#define ends_with startswith
#endif

// Compatibility helper function for creation CompoundStmt.
// Clang 15 and above use a extra param FPFeatures in CompoundStmt::Create.

static inline bool SourceManager_isPointWithin(const SourceManager& SM,
                                               SourceLocation Loc,
                                               SourceLocation B,
                                               SourceLocation E) {
  return SM.isPointWithin(Loc, B, E);
}

#if CLANG_VERSION_MAJOR < 15
#define CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam(Node) /**/
#define CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam1(CS) /**/
#define CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam2(FP) /**/
static inline CompoundStmt* CompoundStmt_Create(
        const ASTContext &Ctx, ArrayRef<Stmt *> Stmts,
        SourceLocation LB, SourceLocation RB)
{
   return CompoundStmt::Create(Ctx, Stmts, LB, RB);
}
#elif CLANG_VERSION_MAJOR >= 15
#define CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam(Node) ,(Node)->getFPFeatures()
#define CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam1(CS) ,(((CS)&&(CS)->hasStoredFPFeatures())?(CS)->getStoredFPFeatures():FPOptionsOverride())
#define CLAD_COMPAT_CLANG15_CompoundStmt_Create_ExtraParam2(FP) ,(FP)
static inline CompoundStmt* CompoundStmt_Create(
        const ASTContext &Ctx, ArrayRef<Stmt *> Stmts,
        FPOptionsOverride FPFeatures,
        SourceLocation LB, SourceLocation RB)
{
   return CompoundStmt::Create(Ctx, Stmts, FPFeatures, LB, RB);
}
#endif

#if CLANG_VERSION_MAJOR < 16
static inline NamespaceDecl*
NamespaceDecl_Create(ASTContext& C, DeclContext* DC, bool Inline,
                     SourceLocation StartLoc, SourceLocation IdLoc,
                     IdentifierInfo* Id, NamespaceDecl* PrevDecl) {
   return NamespaceDecl::Create(C, DC, Inline, StartLoc, IdLoc, Id, PrevDecl);
}
#else
static inline NamespaceDecl*
NamespaceDecl_Create(ASTContext& C, DeclContext* DC, bool Inline,
                     SourceLocation StartLoc, SourceLocation IdLoc,
                     IdentifierInfo* Id, NamespaceDecl* PrevDecl) {
   return NamespaceDecl::Create(C, DC, Inline, StartLoc, IdLoc, Id, PrevDecl,
                                /*Nested=*/false);
}
#endif

// Clang 12: bool Expr::EvaluateAsConstantExpr(EvalResult &Result,
// ConstExprUsage Usage, ASTContext &)
// => bool Expr::EvaluateAsConstantExpr(EvalResult &Result, ASTContext &)

static inline bool Expr_EvaluateAsConstantExpr(const Expr* E,
                                               Expr::EvalResult& res,
                                               const ASTContext& Ctx) {
#if CLANG_VERSION_MAJOR < 12
  return E->EvaluateAsConstantExpr(res, Expr::EvaluateForCodeGen, Ctx);
#else
  return E->EvaluateAsConstantExpr(res, Ctx);
#endif
}

// Compatibility helper function for creation IfStmt.
// Clang 12 and above use two extra params.

static inline IfStmt* IfStmt_Create(const ASTContext &Ctx,
   SourceLocation IL, bool IsConstexpr,
   Stmt *Init, VarDecl *Var, Expr *Cond,
   SourceLocation LPL, SourceLocation RPL,
   Stmt *Then, SourceLocation EL=SourceLocation(), Stmt *Else=nullptr)
{

#if CLANG_VERSION_MAJOR < 12
  return IfStmt::Create(Ctx, IL, IsConstexpr, Init, Var, Cond, Then, EL, Else);
#elif CLANG_VERSION_MAJOR < 14
  return IfStmt::Create(Ctx, IL, IsConstexpr, Init, Var, Cond, LPL, RPL, Then,
                        EL, Else);
#elif CLANG_VERSION_MAJOR >= 14
   IfStatementKind kind = IfStatementKind::Ordinary;
   if (IsConstexpr)
      kind = IfStatementKind::Constexpr;
   return IfStmt::Create(Ctx, IL, kind, Init, Var, Cond, LPL, RPL, Then, EL, Else);   
#endif
}

// Compatibility helper function for creation CallExpr.
// Clang 12 and above use one extra param.

#if CLANG_VERSION_MAJOR < 12
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

// Clang 12 and above use one extra param.

#if CLANG_VERSION_MAJOR < 12
#define CLAD_COMPAT_CLANG8_CallExpr_ExtraParams                                \
  , Node->getNumArgs(), Node->getADLCallKind()
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
   #if CLANG_VERSION_MAJOR >= 16
      #define CLAD_COMPAT_CLANG11_LangOptions_EtraParams /**/
   #else
      #define CLAD_COMPAT_CLANG11_LangOptions_EtraParams Ctx.getLangOpts()
   #endif
   #define CLAD_COMPAT_CLANG11_Ctx_ExtraParams Ctx,
   #define CLAD_COMPAT_CREATE11(CLASS, CTORARGS) (CLASS::Create CTORARGS)
   #define CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Removed /**/
   #define CLAD_COMPAT_CLANG11_CompoundAssignOperator_EtraParams_Moved ,Node->getComputationLHSType(),Node->getComputationResultType()
   #define CLAD_COMPAT_CLANG11_ChooseExpr_EtraParams_Removed /**/
   #define CLAD_COMPAT_CLANG11_WhileStmt_ExtraParams ,Node->getLParenLoc(),Node->getRParenLoc()
#endif

// Compatibility helper function for creation CXXMemberCallExpr.
// Clang 12 and above use two extra param.

#if CLANG_VERSION_MAJOR < 12
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

// Compatibility helper function for creation SwitchStmt.
// Clang 12 and above use two extra params.

static inline SwitchStmt* SwitchStmt_Create(const ASTContext &Ctx,
   Stmt *Init, VarDecl *Var, Expr *Cond,
   SourceLocation LParenLoc, SourceLocation RParenLoc)
{

#if CLANG_VERSION_MAJOR < 12
  return SwitchStmt::Create(Ctx, Init, Var, Cond);
#elif CLANG_VERSION_MAJOR >= 12
   return SwitchStmt::Create(Ctx, Init, Var, Cond, LParenLoc, RParenLoc);
#endif
}

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
// clang 18 clang::ArrayType::ArraySizeModifier became clang::ArraySizeModifier
#if CLANG_VERSION_MAJOR < 18
static inline QualType
getConstantArrayType(const ASTContext& Ctx, QualType EltTy,
                     const APInt& ArySize, const Expr* SizeExpr,
                     clang::ArrayType::ArraySizeModifier ASM,
                     unsigned IndexTypeQuals) {
#if CLANG_VERSION_MAJOR < 10
   return Ctx.getConstantArrayType(EltTy, ArySize, ASM, IndexTypeQuals);
#elif CLANG_VERSION_MAJOR >= 10
  return Ctx.getConstantArrayType(EltTy, ArySize, SizeExpr, ASM,
                                  IndexTypeQuals);
#endif
}
#else
static inline QualType
getConstantArrayType(const ASTContext& Ctx, QualType EltTy,
                     const APInt& ArySize, const Expr* SizeExpr,
                     clang::ArraySizeModifier ASM, unsigned IndexTypeQuals) {
  return Ctx.getConstantArrayType(EltTy, ArySize, SizeExpr, ASM,
                                  IndexTypeQuals);
}
#endif

// Clang 10 add new last param TrailingRequiresClause in FunctionDecl::Create

#if CLANG_VERSION_MAJOR < 10
   #define CLAD_COMPAT_CLANG10_FunctionDecl_Create_ExtraParams(x) /**/
#elif CLANG_VERSION_MAJOR >= 10
#define CLAD_COMPAT_CLANG10_FunctionDecl_Create_ExtraParams(x)                 \
  , ((x) ? VB.Clone((x)) : nullptr)
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
// Clang 15 add one extra param to clang::Declarator() - const ParsedAttributesView & DeclarationAttrs

#if CLANG_VERSION_MAJOR < 12
   #define CLAD_COMPAT_CLANG12_Declarator_LambdaExpr clang::DeclaratorContext::LambdaExprContext
   #define CLAD_COMPAT_CLANG15_Declarator_DeclarationAttrs_ExtraParam /**/
#elif CLANG_VERSION_MAJOR < 15
   #define CLAD_COMPAT_CLANG12_Declarator_LambdaExpr clang::DeclaratorContext::LambdaExpr
   #define CLAD_COMPAT_CLANG15_Declarator_DeclarationAttrs_ExtraParam /**/
#elif CLANG_VERSION_MAJOR >= 15
   #define CLAD_COMPAT_CLANG12_Declarator_LambdaExpr clang::DeclaratorContext::LambdaExpr
   #define CLAD_COMPAT_CLANG15_Declarator_DeclarationAttrs_ExtraParam clang::ParsedAttributesView::none(),
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

/// Clang < 9, do not provide `Sema::BuildCXXThisExpr` function.
static inline CXXThisExpr* Sema_BuildCXXThisExpr(Sema& SemaRef,
                                                 const CXXMethodDecl* method) {
  auto thisType = method->getThisType();
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

#if CLANG_VERSION_MAJOR > 8
static inline Qualifiers CXXMethodDecl_getMethodQualifiers(const CXXMethodDecl* MD) {
   return MD->getMethodQualifiers();
}
#elif CLANG_VERSION_MAJOR == 8
static inline Qualifiers CXXMethodDecl_getMethodQualifiers(const CXXMethodDecl* MD) {
   return MD->getTypeQualifiers();
}
#endif

// Clone declarations. `ValueStmt` node is only available after clang 8.
#if CLANG_VERSION_MAJOR <= 8
#define CLAD_COMPAT_8_DECLARE_CLONE_FN(ValueStmt) /**/
#elif CLANG_VERSION_MAJOR > 8
#define CLAD_COMPAT_8_DECLARE_CLONE_FN(ValueStmt) DECLARE_CLONE_FN(ValueStmt)
#endif

#if CLANG_VERSION_MAJOR <= 13
#define CLAD_COMPAT_FunctionDecl_UsesFPIntrin_Param(FD) /**/
#elif CLANG_VERSION_MAJOR > 13
#define CLAD_COMPAT_FunctionDecl_UsesFPIntrin_Param(FD) , FD->UsesFPIntrin()
#endif

#if CLANG_VERSION_MAJOR <= 13
#define CLAD_COMPAT_IfStmt_Create_IfStmtKind_Param(Node) Node->isConstexpr()
#elif CLANG_VERSION_MAJOR > 13
#define CLAD_COMPAT_IfStmt_Create_IfStmtKind_Param(Node) Node->getStatementKind()
#endif

#if CLANG_VERSION_MAJOR < 9
static inline MemberExpr* BuildMemberExpr(
    Sema& semaRef, Expr* base, bool isArrow, SourceLocation opLoc,
    const CXXScopeSpec* SS, SourceLocation templateKWLoc, ValueDecl* member,
    DeclAccessPair foundDecl, bool hadMultipleCandidates,
    const DeclarationNameInfo& memberNameInfo, QualType ty, ExprValueKind VK,
    ExprObjectKind OK, const TemplateArgumentListInfo* templateArgs = nullptr) {
  auto& C = semaRef.getASTContext();
  auto NNSLoc = SS->getWithLocInContext(C);
  return MemberExpr::Create(C, base, isArrow, opLoc, NNSLoc, templateKWLoc,
                            member, foundDecl, memberNameInfo, templateArgs, ty,
                            VK, OK);
}
#else
static inline MemberExpr* BuildMemberExpr(
    Sema& semaRef, Expr* base, bool isArrow, SourceLocation opLoc,
    const CXXScopeSpec* SS, SourceLocation templateKWLoc, ValueDecl* member,
    DeclAccessPair foundDecl, bool hadMultipleCandidates,
    const DeclarationNameInfo& memberNameInfo, QualType ty, ExprValueKind VK,
    ExprObjectKind OK, const TemplateArgumentListInfo* templateArgs = nullptr) {
  return semaRef.BuildMemberExpr(base, isArrow, opLoc, SS, templateKWLoc,
                                 member, foundDecl, hadMultipleCandidates,
                                 memberNameInfo, ty, VK, OK, templateArgs);
}
#endif

#if CLANG_VERSION_MAJOR < 10
static inline Expr* GetSubExpr(const MaterializeTemporaryExpr* MTE) {
  return MTE->GetTemporaryExpr();
}
#else
static inline Expr* GetSubExpr(const MaterializeTemporaryExpr* MTE) {
  return MTE->getSubExpr();
}
#endif

#if CLANG_VERSION_MAJOR < 9
static inline QualType
CXXMethodDecl_GetThisObjectType(Sema& semaRef, const CXXMethodDecl* MD) {
  ASTContext& C = semaRef.getASTContext();
  const CXXRecordDecl* RD = MD->getParent();
  auto RDType = RD->getTypeForDecl();
  auto thisObjectQType = C.getQualifiedType(
      RDType, clad_compat::CXXMethodDecl_getMethodQualifiers(MD));
  return thisObjectQType;
}
#else
static inline QualType
CXXMethodDecl_GetThisObjectType(Sema& semaRef, const CXXMethodDecl* MD) {
// clang-18 renamed getThisObjectType to getFunctionObjectParameterType
#if CLANG_VERSION_MAJOR < 18
  return MD->getThisObjectType();
#else
  return MD->getFunctionObjectParameterType();
#endif
}
#endif

#if CLANG_VERSION_MAJOR < 12
#define CLAD_COMPAT_SubstNonTypeTemplateParmExpr_isReferenceParameter_ExtraParam( \
    Node) /**/
#else
#define CLAD_COMPAT_SubstNonTypeTemplateParmExpr_isReferenceParameter_ExtraParam( \
    Node)                                                                         \
  Node->isReferenceParameter(),
#endif

#if CLANG_VERSION_MAJOR < 16
template <typename T>
llvm::ArrayRef<T> makeArrayRef(const T* data, size_t length) {
  return llvm::makeArrayRef(data, length);
}
template <typename T>
llvm::ArrayRef<T> makeArrayRef(const T* begin, const T* end) {
  return llvm::makeArrayRef(begin, end);
}
#else
template <typename T>
llvm::ArrayRef<T> makeArrayRef(const T* data, size_t length) {
  return llvm::ArrayRef(data, length);
}
template <typename T>
llvm::ArrayRef<T> makeArrayRef(const T* begin, const T* end) {
  return llvm::ArrayRef(begin, end);
}
#endif

#if CLANG_VERSION_MAJOR < 16
template <typename T> using llvm_Optional = llvm::Optional<T>;
#else
template <typename T> using llvm_Optional = std::optional<T>;
#endif

#if CLANG_VERSION_MAJOR < 16
template <typename T> T& llvm_Optional_GetValue(llvm::Optional<T>& opt) {
  return opt.getValue();
}
#else
template <typename T> T& llvm_Optional_GetValue(std::optional<T>& opt) {
  return opt.value();
}
#endif

#if CLANG_VERSION_MAJOR < 9
static inline Expr* ArraySize_None() { return nullptr; }
#else
static inline llvm_Optional<Expr*> ArraySize_None() {
  return llvm_Optional<Expr*>();
}
#endif

#if CLANG_VERSION_MAJOR < 9
static inline const Expr* ArraySize_GetValue(const Expr* val) { return val; }
#elif CLANG_VERSION_MAJOR < 16
static inline const Expr*
ArraySize_GetValue(const llvm::Optional<const Expr*>& opt) {
  return opt.getValue();
}
#else
static inline const Expr*
ArraySize_GetValue(const std::optional<const Expr*>& opt) {
   return opt.value();
}
#endif

#if CLANG_VERSION_MAJOR < 13
static inline bool IsPRValue(const Expr* E) { return E->isRValue(); }
#else
static inline bool IsPRValue(const Expr* E) { return E->isPRValue(); }
#endif

#if CLANG_VERSION_MAJOR >= 9
#define CLAD_COMPAT_CLANG9_CXXDefaultArgExpr_getUsedContext_Param(Node)               \
  , Node->getUsedContext()
#else
#define CLAD_COMPAT_CLANG9_CXXDefaultArgExpr_getUsedContext_Param(Node) /**/
#endif

#if CLANG_VERSION_MAJOR >= 16
#define CLAD_COMPAT_CLANG16_CXXDefaultArgExpr_getRewrittenExpr_Param(Node)     \
  , Node->getRewrittenExpr()
#else
#define CLAD_COMPAT_CLANG16_CXXDefaultArgExpr_getRewrittenExpr_Param(Node) /**/
#endif

// Clang 15 renamed StringKind::Ascii to StringKind::Ordinary
// Clang 18 renamed clang::StringLiteral::StringKind::Ordinary became
// clang::StringLiteralKind::Ordinary;

#if CLANG_VERSION_MAJOR < 15
const auto StringLiteralKind_Ordinary = clang::StringLiteral::StringKind::Ascii;
#elif CLANG_VERSION_MAJOR >= 15
#if CLANG_VERSION_MAJOR < 18
const auto StringLiteralKind_Ordinary =
    clang::StringLiteral::StringKind::Ordinary;
#else
const auto StringLiteralKind_Ordinary = clang::StringLiteralKind::Ordinary;
#endif
#endif

// Clang 15 add one extra param to Sema::CheckFunctionDeclaration

#if CLANG_VERSION_MAJOR < 15
#define CLAD_COMPAT_CheckFunctionDeclaration_DeclIsDefn_ExtraParam(Fn) /**/
#elif CLANG_VERSION_MAJOR >= 15
#define CLAD_COMPAT_CheckFunctionDeclaration_DeclIsDefn_ExtraParam(Fn) \
  ,Fn->isThisDeclarationADefinition()
#endif


// Clang 17 change type of last param of ActOnStartOfLambdaDefinition
// from Scope* to 'const DeclSpec&'
#if CLANG_VERSION_MAJOR < 17
static inline Scope* Sema_ActOnStartOfLambdaDefinition_ScopeOrDeclSpec(Scope *CurScope, const DeclSpec &DS) {
  return CurScope;
}
#elif CLANG_VERSION_MAJOR >= 17
static inline const DeclSpec& Sema_ActOnStartOfLambdaDefinition_ScopeOrDeclSpec(Scope *CurScope, const DeclSpec &DS) {
  return DS;
}
#endif

// Clang 17 add one extra param to clang::PredefinedExpr::Create - isTransparent

#if CLANG_VERSION_MAJOR < 17
#define CLAD_COMPAT_CLANG17_IsTransparent(Node) /**/
#elif CLANG_VERSION_MAJOR >= 17
#define CLAD_COMPAT_CLANG17_IsTransparent(Node) \
  ,Node->isTransparent()
#endif

// Clang 9 above added isa_and_nonnull.
#if CLANG_VERSION_MAJOR < 9
template <typename X, typename Y> bool isa_and_nonnull(const Y* Val) {
  return Val && isa<X>(Val);
}
#else
template <typename X, typename Y> bool isa_and_nonnull(const Y* Val) {
  return llvm::isa_and_nonnull<X>(Val);
}
#endif

} // namespace clad_compat
#endif //CLAD_COMPATIBILITY
