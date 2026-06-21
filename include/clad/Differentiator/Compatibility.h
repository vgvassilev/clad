//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// author:  Alexander Penev <alexander_penev@yahoo.com>
//------------------------------------------------------------------------------

#ifndef CLAD_COMPATIBILITY
#define CLAD_COMPATIBILITY

#include "llvm/Config/llvm-config.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/AST/TemplateName.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeLoc.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Basic/Version.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Ownership.h"
#include "clang/Sema/Sema.h"
#if CLANG_VERSION_MAJOR > 18
#include "clang/Sema/SemaOpenMP.h"
#endif

namespace clad_compat {

using namespace clang;
using namespace llvm;

// clang-21
#if CLANG_VERSION_MAJOR < 21
#define CLAD_COMPAT_CLANG21_AtEndOfTUParam
#define CLAD_COMPAT_CLANG21_CSSExtendKWLocExtraParam(V) (V),
#define CLAD_COMPAT_CLANG21_StringLiteralParams(V) &(V)[0], (V).size()
#define CLAD_COMPAT_CLANG21_StringLiteralParamsRange , SourceRange()
#define CLAD_COMPAT_CLANG21_getTrailingRequiresClause(FD)                      \
  const_cast<FunctionDecl*>(FD)->getTrailingRequiresClause()
#define CLAD_COMPAT_CLANG21_getTrailingRequiresExpr(FD)                        \
  (FD)->getTrailingRequiresClause()
#define CLAD_COMPAT_CLANG21_UpdateTrailingRequiresClause(Trailing, V)          \
  (Trailing) = (V)
#define CLAD_COMPAT_CLANG21_TemplateKeywordParam , /*TemplateKeyword=*/false
#else
#define CLAD_COMPAT_CLANG21_AtEndOfTUParam , /*AtEndOfTU=*/true
#define CLAD_COMPAT_CLANG21_CSSExtendKWLocExtraParam(V)
#define CLAD_COMPAT_CLANG21_StringLiteralParams(V) (V)
#define CLAD_COMPAT_CLANG21_StringLiteralParamsRange
#define CLAD_COMPAT_CLANG21_getTrailingRequiresClause(FD)                      \
  (FD)->getTrailingRequiresClause()
#define CLAD_COMPAT_CLANG21_getTrailingRequiresExpr(FD)                        \
  (FD)->getTrailingRequiresClause().ConstraintExpr
#define CLAD_COMPAT_CLANG21_UpdateTrailingRequiresClause(Trailing, V)          \
  (Trailing).ConstraintExpr = (V)
#define CLAD_COMPAT_CLANG21_TemplateKeywordParam
#endif

// clang-21 OpenMPReductionClauseModifiers  got extra argument
#if CLANG_VERSION_MAJOR < 21
#define CLAD_COMPAT_CLANG21_getModifier(Clause) (Clause)->getModifier()
#define CLAD_COMPAT_CLANG21_createModifier(Modifier) (Modifier)
#else
#define CLAD_COMPAT_CLANG21_getModifier(Clause)                                \
  {(Clause)->getModifier(), (Clause)->getOriginalSharingModifier()}
#define CLAD_COMPAT_CLANG21_createModifier(Modifier)                           \
  SemaOpenMP::OpenMPVarListDataTy::OpenMPReductionClauseModifiers(             \
      Modifier, OMPC_ORIGINAL_SHARING_default)
#endif

// clang-20 Clause varlist typo
#if CLANG_VERSION_MAJOR < 20
#define CLAD_COMPAT_CLANG20_getvarlist(Clause) (Clause)->varlists()
#else
#define CLAD_COMPAT_CLANG20_getvarlist(Clause) (Clause)->varlist()
#endif

// clang-20 clang::Sema::AA_Casting became scoped
#if CLANG_VERSION_MAJOR < 20
#define CLAD_COMPAT_CLANG20_SemaAACasting clang::Sema::AA_Casting
#else
#define CLAD_COMPAT_CLANG20_SemaAACasting clang::AssignmentAction::Casting
#endif

// clang-19 SemaOpenMP was introduced
#if CLANG_VERSION_MAJOR < 19
#define CLAD_COMPAT_CLANG19_SemaOpenMP(Sema) (Sema)
#else
#define CLAD_COMPAT_CLANG19_SemaOpenMP(Sema) ((Sema).OpenMP())
#endif

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

#elif CLANG_VERSION_MAJOR == 18
  bool KnownDependent = false;
  return UnresolvedLookupExpr::Create(Ctx, NamingClass, QualifierLoc,
                                      TemplateKWLoc, NameInfo, RequiresADL,
                                      // They get copied again by
                                      // OverloadExpr, so we are safe.
                                      Args, Begin, End, KnownDependent);
#else
  bool KnownDependent = false;
  bool KnownInstantiationDependent = false;
  return UnresolvedLookupExpr::Create(
      Ctx, NamingClass, QualifierLoc, TemplateKWLoc, NameInfo, RequiresADL,
      // They get copied again by
      // OverloadExpr, so we are safe.
      Args, Begin, End, KnownDependent, KnownInstantiationDependent);
#endif
}

// Clang 18 ETK_None -> ElaboratedTypeKeyword::None

#if LLVM_VERSION_MAJOR < 18
const auto ElaboratedTypeKeyword_None = ETK_None;
#else
const auto ElaboratedTypeKeyword_None = ElaboratedTypeKeyword::None;
#endif

// Clang 22 reshaped the type-qualifier model (llvm/llvm-project#147835):
// NestedNameSpecifier became a uintptr value type, ElaboratedType
// folded into TagType, and several ASTContext::get* / TagDecl helpers
// were renamed or removed. Wrappers below isolate the call sites.

#if CLANG_VERSION_MAJOR < 22
using NestedNameSpecifierTy = clang::NestedNameSpecifier*;

// Default "no qualifier" NNS. On LLVM 22 a value-NNS `{}` evaluates
// TRUE (Invalid kind), so the pre-22 `NNS = nullptr` idiom needs this
// wrapper to keep the same truthiness across versions.
inline NestedNameSpecifierTy nullNNS() { return nullptr; }

// True iff NNS actually names a qualifier. Pre-22 a pointer-NNS is
// "no qualifier" iff null; mirrors `if (!NNS)`.
inline bool hasQualifier(NestedNameSpecifierTy NNS) { return NNS != nullptr; }

// Wrap: does QT already carry namespace elaboration? Pre-22 stored
// the elaboration in a separate ElaboratedType wrapper.
inline bool isElaboratedType(clang::QualType QT) {
  return QT->getAs<clang::ElaboratedType>() != nullptr;
}

// Pre-22 getPrefix() worked uniformly on any NNS kind.
inline bool hasNNSPrefix(clang::NestedNameSpecifier* NS) {
  return NS && NS->getPrefix() != nullptr;
}
inline clang::NestedNameSpecifier*
getNNSPrefix(clang::NestedNameSpecifier* NS) {
  return NS->getPrefix();
}

inline clang::QualType getElaboratedType(clang::ASTContext& C,
                                         clang::ElaboratedTypeKeyword Keyword,
                                         clang::NestedNameSpecifier* Qualifier,
                                         clang::QualType QT) {
  return C.getElaboratedType(Keyword, Qualifier, QT);
}

inline clang::QualType getRecordType(clang::ASTContext& C,
                                     const clang::RecordDecl* RD) {
  return C.getRecordType(RD);
}

inline clang::QualType getCanonicalTagType(clang::ASTContext& /*C*/,
                                           const clang::TagDecl* TD) {
  return TD->getTypeForDecl()->getCanonicalTypeInternal();
}

inline clang::QualType
CheckTemplateIdType(clang::Sema& S, clang::TemplateName T,
                    clang::SourceLocation Loc,
                    clang::TemplateArgumentListInfo& TLI) {
  return S.CheckTemplateIdType(T, Loc, TLI);
}
#else // CLANG_VERSION_MAJOR >= 22
using NestedNameSpecifierTy = clang::NestedNameSpecifier;

inline NestedNameSpecifierTy nullNNS() {
  return clang::NestedNameSpecifier(std::nullopt);
}

// True iff NNS names a qualifier. LLVM 22's value-NNS distinguishes
// Null from Invalid; both mean "no qualifier" to clad, mirroring the
// pre-22 `if (!NNS)` idiom.
inline bool hasQualifier(NestedNameSpecifierTy NNS) {
  return bool(NNS) && NNS.getKind() != clang::NestedNameSpecifier::Kind::Null;
}

// Clang 22: TagType (and friends) carry keyword + qualifier inline.
// "Already elaborated" becomes "the tag carries a non-default keyword
// or a qualifier"; non-tag types are not elaboration carriers.
inline bool isElaboratedType(clang::QualType QT) {
  if (const auto* TT = QT->getAs<clang::TagType>())
    return TT->getKeyword() != clang::ElaboratedTypeKeyword::None ||
           bool(TT->getQualifier());
  return false;
}

// Clang 22: only Namespace-kind NNS exposes a prefix; other kinds
// have no analogue to the old uniform getPrefix(), so treat them as
// "no prefix" -- the pre-22 callers branched on null-prefix anyway.
inline bool hasNNSPrefix(clang::NestedNameSpecifier NS) {
  if (NS.getKind() == clang::NestedNameSpecifier::Kind::Namespace)
    return bool(NS.getAsNamespaceAndPrefix().Prefix);
  return false;
}
inline clang::NestedNameSpecifier getNNSPrefix(clang::NestedNameSpecifier NS) {
  if (NS.getKind() == clang::NestedNameSpecifier::Kind::Namespace)
    return NS.getAsNamespaceAndPrefix().Prefix;
  // Null (std::nullopt), not the default ctor's FlagKind::Invalid, which
  // would later UNREACHABLE in getKind() if anyone inspected it.
  return std::nullopt;
}

// Clang 22 (llvm/llvm-project#147835): keyword + qualifier ride inline
// on the type, so getElaboratedType is gone -- rebuild via the
// type-specific factory. Branches cover record/enum, typedef, and
// template-specialization types.
inline clang::QualType getElaboratedType(clang::ASTContext& C,
                                         clang::ElaboratedTypeKeyword Keyword,
                                         clang::NestedNameSpecifier Qualifier,
                                         clang::QualType QT) {
  // Alias-template TSTs (e.g. `clad::tape<T>`) must stay TSTs so the
  // alias name survives; routing them through getTagType would resolve
  // `tape` to its underlying `tape_impl` record.
  if (const auto* TST = QT->getAs<clang::TemplateSpecializationType>();
      TST && TST->isTypeAlias()) {
    clang::TemplateName QTN = C.getQualifiedTemplateName(
        Qualifier, /*TemplateKeyword=*/false, TST->getTemplateName());
    return C.getTemplateSpecializationType(
        Keyword, QTN, TST->template_arguments(),
        /*CanonicalArgs=*/{}, TST->getAliasedType());
  }
  if (const auto* TT = QT->getAs<clang::TagType>())
    return C.getTagType(Keyword, Qualifier, TT->getDecl(), /*OwnsTag=*/false);
  if (const auto* TyD = QT->getAs<clang::TypedefType>())
    return C.getTypedefType(Keyword, Qualifier, TyD->getDecl());
  if (const auto* TST = QT->getAs<clang::TemplateSpecializationType>()) {
    clang::TemplateName QTN = C.getQualifiedTemplateName(
        Qualifier, /*TemplateKeyword=*/false, TST->getTemplateName());
    return C.getTemplateSpecializationType(Keyword, QTN,
                                           TST->template_arguments(),
                                           /*CanonicalArgs=*/{}, QualType());
  }
  return QT;
}

// Clang 22: getRecordType -> getTagType with default keyword/qualifier.
inline clang::QualType getRecordType(clang::ASTContext& C,
                                     const clang::RecordDecl* RD) {
  return C.getTagType(clang::ElaboratedTypeKeyword::None, std::nullopt, RD,
                      /*OwnsTag=*/false);
}

// Clang 22: TagDecl::getTypeForDecl() was deleted; use
// ASTContext::getCanonicalTagType(TD) instead.
inline clang::QualType getCanonicalTagType(clang::ASTContext& C,
                                           const clang::TagDecl* TD) {
  return C.getCanonicalTagType(TD);
}

// Clang 22: CheckTemplateIdType gained Keyword + Scope* +
// ForNestedNameSpecifier params; default them to match pre-22 behavior.
inline clang::QualType
CheckTemplateIdType(clang::Sema& S, clang::TemplateName T,
                    clang::SourceLocation Loc,
                    clang::TemplateArgumentListInfo& TLI) {
  return S.CheckTemplateIdType(clang::ElaboratedTypeKeyword::None, T, Loc, TLI,
                               /*Scope=*/nullptr,
                               /*ForNestedNameSpecifier=*/false);
}
#endif

// Clang 22 (llvm/llvm-project#147835): NestedNameSpecifier::Create
// factories are gone -- value-NNS constructors take their place, and
// the type-kind no longer accepts a prefix. The wrappers bake the
// prefix into the type at construction so it rides transitively.
inline NestedNameSpecifierTy makeNNSNamespace(clang::ASTContext& C,
                                              NestedNameSpecifierTy Prefix,
                                              const clang::NamespaceDecl* NS) {
#if CLANG_VERSION_MAJOR < 22
  return clang::NestedNameSpecifier::Create(C, Prefix, NS);
#else
  return clang::NestedNameSpecifier(C, NS, Prefix);
#endif
}

// Build a NestedNameSpecifier of the form "Prefix::TD::". Pre-22 took
// Prefix as a separate Create() arg; LLVM 22 bakes Prefix into the
// TagType via getTagType so the value-NNS retains the full path.
inline NestedNameSpecifierTy makeNNSTagType(clang::ASTContext& C,
                                            NestedNameSpecifierTy Prefix,
                                            const clang::TagDecl* TD) {
#if CLANG_VERSION_MAJOR < 22
  clang::QualType T = C.getTypeDeclType(TD);
  return clang::NestedNameSpecifier::Create(
      C, Prefix CLAD_COMPAT_CLANG21_TemplateKeywordParam, T.getTypePtr());
#else
  clang::QualType T = C.getTagType(clang::ElaboratedTypeKeyword::None, Prefix,
                                   TD, /*OwnsTag=*/false);
  return clang::NestedNameSpecifier(T.getTypePtr());
#endif
}

// Clang 22: ASTContext::getTypeDeclType(TypeDecl*) was deleted in the
// 1-arg form; the 3-arg form now requires Keyword + Qualifier.
inline clang::QualType getTypeDeclType(clang::ASTContext& C,
                                       const clang::TypeDecl* TD) {
#if CLANG_VERSION_MAJOR < 22
  return C.getTypeDeclType(TD);
#else
  return C.getTypeDeclType(clang::ElaboratedTypeKeyword::None, std::nullopt,
                           TD);
#endif
}

// Clang 22 (llvm/llvm-project#143653): getSizeType returns a
// PredefinedSugarType that prints as "__size_t". clad's tests want
// the platform integer spelling -- use the canonical type.
inline clang::QualType getSizeType(clang::ASTContext& C) {
#if CLANG_VERSION_MAJOR < 22
  return C.getSizeType();
#else
  return C.getCanonicalSizeType();
#endif
}

// LLVM 22 PredefinedSugarType (size_t/ptrdiff_t/...) leaks the
// "__size_t" name into generated temporaries; strip to the canonical
// integer at the declaration point.
inline clang::QualType stripPredefinedSugar(clang::QualType QT) {
#if CLANG_VERSION_MAJOR >= 22
  if (clang::isa<clang::PredefinedSugarType>(QT.getTypePtr()))
    return QT.getCanonicalType();
#endif
  return QT;
}

// Clang 22: ActOnBreakStmt / ActOnContinueStmt gained named-target
// params (Label + LabelLoc). Default them for unnamed break/continue.
inline clang::StmtResult ActOnBreakStmt(clang::Sema& S,
                                        clang::SourceLocation Loc,
                                        clang::Scope* CurScope) {
#if CLANG_VERSION_MAJOR < 22
  return S.ActOnBreakStmt(Loc, CurScope);
#else
  return S.ActOnBreakStmt(Loc, CurScope, /*Label=*/nullptr,
                          /*LabelLoc=*/clang::SourceLocation());
#endif
}
inline clang::StmtResult ActOnContinueStmt(clang::Sema& S,
                                           clang::SourceLocation Loc,
                                           clang::Scope* CurScope) {
#if CLANG_VERSION_MAJOR < 22
  return S.ActOnContinueStmt(Loc, CurScope);
#else
  return S.ActOnContinueStmt(Loc, CurScope, /*Label=*/nullptr,
                             /*LabelLoc=*/clang::SourceLocation());
#endif
}

// Clang 22: BreakStmt/ContinueStmt share LoopControlStmt::getKwLoc();
// pre-22 had stmt-specific getBreakLoc/getContinueLoc.
inline clang::SourceLocation getBreakLoc(const clang::BreakStmt* BS) {
#if CLANG_VERSION_MAJOR < 22
  return BS->getBreakLoc();
#else
  return BS->getKwLoc();
#endif
}
inline clang::SourceLocation getContinueLoc(const clang::ContinueStmt* CS) {
#if CLANG_VERSION_MAJOR < 22
  return CS->getContinueLoc();
#else
  return CS->getKwLoc();
#endif
}

// CXXScopeSpec::Extend's TypeLoc overload was removed in clang 22;
// replacement is build-an-NNS + MakeTrivial. clang 20+ keeps the
// 3-arg form; clang <20 used a leading SourceLocation keyword-loc
// param. The wrapper hides all three call shapes.
inline void CSS_ExtendType(clang::CXXScopeSpec& CSS, clang::ASTContext& C,
                           clang::SourceLocation KWLoc, clang::TypeLoc TL,
                           clang::SourceLocation ColonColonLoc) {
#if CLANG_VERSION_MAJOR >= 22
  clang::NestedNameSpecifier NNS(TL.getType().getTypePtr());
  CSS.MakeTrivial(C, NNS, clang::SourceRange(TL.getBeginLoc(), ColonColonLoc));
#elif CLANG_VERSION_MAJOR >= 21
  (void)KWLoc;
  CSS.Extend(C, TL, ColonColonLoc);
#else
  CSS.Extend(C, KWLoc, TL, ColonColonLoc);
#endif
}

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

// Compatibility helper function for creation IfStmt.
// Clang 14 switched from bool IsConstexpr to IfStatementKind.

static inline IfStmt* IfStmt_Create(const ASTContext &Ctx,
   SourceLocation IL, bool IsConstexpr,
   Stmt *Init, VarDecl *Var, Expr *Cond,
   SourceLocation LPL, SourceLocation RPL,
   Stmt *Then, SourceLocation EL=SourceLocation(), Stmt *Else=nullptr)
{

#if CLANG_VERSION_MAJOR < 14
  return IfStmt::Create(Ctx, IL, IsConstexpr, Init, Var, Cond, LPL, RPL, Then,
                        EL, Else);
#else
  IfStatementKind kind = IfStatementKind::Ordinary;
  if (IsConstexpr)
    kind = IfStatementKind::Constexpr;
  return IfStmt::Create(Ctx, IL, kind, Init, Var, Cond, LPL, RPL, Then, EL,
                        Else);
#endif
}

// Compatibility helper function for creation CallExpr and CUDAKernelCallExpr.

static inline CallExpr* CallExpr_Create(const ASTContext &Ctx, Expr *Fn, ArrayRef< Expr *> Args,
   QualType Ty, ExprValueKind VK, SourceLocation RParenLoc, FPOptionsOverride FPFeatures,
   unsigned MinNumArgs = 0, CallExpr::ADLCallKind UsesADL = CallExpr::NotADL)
{
   return CallExpr::Create(Ctx, Fn, Args, Ty, VK, RParenLoc, FPFeatures, MinNumArgs, UsesADL);
}

static inline CUDAKernelCallExpr*
CUDAKernelCallExpr_Create(const ASTContext& Ctx, Expr* Fn, CallExpr* Config,
                          ArrayRef<Expr*> Args, QualType Ty, ExprValueKind VK,
                          SourceLocation RParenLoc,
                          FPOptionsOverride FPFeatures, unsigned MinNumArgs = 0,
                          CallExpr::ADLCallKind UsesADL = CallExpr::NotADL) {
  return CUDAKernelCallExpr::Create(Ctx, Fn, Config, Args, Ty, VK, RParenLoc,
                                    FPFeatures, MinNumArgs);
}

static inline void ExprSetDeps(Expr* result, Expr* Node) {
  struct ExprDependenceAccessor : public Expr {
    void setDependence(ExprDependence Deps) { Expr::setDependence(Deps); }
  };
   ((ExprDependenceAccessor*)result)->setDependence(Node->getDependence());
}

template<class T>
static inline T GetResult(ActionResult<T> Res)
{
   return Res.get();
}

// clang 18 clang::ArrayType::ArraySizeModifier became clang::ArraySizeModifier
#if CLANG_VERSION_MAJOR < 18
static inline QualType
getConstantArrayType(const ASTContext& Ctx, QualType EltTy,
                     const APInt& ArySize, const Expr* SizeExpr,
                     clang::ArrayType::ArraySizeModifier ASM,
                     unsigned IndexTypeQuals) {
  return Ctx.getConstantArrayType(EltTy, ArySize, SizeExpr, ASM,
                                  IndexTypeQuals);
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

// Clang 15 added one extra param to clang::Declarator() --
// const ParsedAttributesView & DeclarationAttrs.
#if CLANG_VERSION_MAJOR < 15
#define CLAD_COMPAT_CLANG15_Declarator_DeclarationAttrs_ExtraParam /**/
#else
#define CLAD_COMPAT_CLANG15_Declarator_DeclarationAttrs_ExtraParam             \
  clang::ParsedAttributesView::none(),
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

#if CLANG_VERSION_MAJOR < 19
#define CLAD_COMPAT_Sema_ForVisibleRedeclaration Sema::ForVisibleRedeclaration
#else
#define CLAD_COMPAT_Sema_ForVisibleRedeclaration                               \
  RedeclarationKind::ForVisibleRedeclaration
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

#if CLANG_VERSION_MAJOR < 19
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
#else
static inline MemberExpr* BuildMemberExpr(
    Sema& semaRef, Expr* base, bool isArrow, SourceLocation opLoc,
    const CXXScopeSpec* SS, SourceLocation templateKWLoc, ValueDecl* member,
    DeclAccessPair foundDecl, bool hadMultipleCandidates,
    const DeclarationNameInfo& memberNameInfo, QualType ty, ExprValueKind VK,
    ExprObjectKind OK, const TemplateArgumentListInfo* templateArgs = nullptr) {
  NestedNameSpecifierLoc NNS =
      SS ? SS->getWithLocInContext(semaRef.getASTContext())
         : NestedNameSpecifierLoc();
  return semaRef.BuildMemberExpr(base, isArrow, opLoc, NNS, templateKWLoc,
                                 member, foundDecl, hadMultipleCandidates,
                                 memberNameInfo, ty, VK, OK, templateArgs);
}
#endif

static inline QualType
CXXMethodDecl_GetThisObjectType(Sema& semaRef, const CXXMethodDecl* MD) {
// clang-18 renamed getThisObjectType to getFunctionObjectParameterType
#if CLANG_VERSION_MAJOR < 18
  return MD->getThisObjectType();
#else
  return MD->getFunctionObjectParameterType();
#endif
}

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

#if CLANG_VERSION_MAJOR < 16
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

#if CLANG_VERSION_MAJOR < 16
#define CLAD_COMPAT_CLANG16_LangOptions_ExtraParams Ctx.getLangOpts()
#else
#define CLAD_COMPAT_CLANG16_LangOptions_ExtraParams /**/
#endif

#if CLANG_VERSION_MAJOR < 13
static inline bool IsPRValue(const Expr* E) { return E->isRValue(); }
#else
static inline bool IsPRValue(const Expr* E) { return E->isPRValue(); }
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

// Clang 19 renamed the enum representing template resolution results
#if CLANG_VERSION_MAJOR >= 19
#define CLAD_COMPAT_TemplateSuccess TemplateDeductionResult::Success
#else
#define CLAD_COMPAT_TemplateSuccess Sema::TDK_Success
#endif

} // namespace clad_compat
#endif //CLAD_COMPATIBILITY
