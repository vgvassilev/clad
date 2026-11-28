//--------------------------------------------------------------------*- C++ -*-
// clad - the C++ Clang-based Automatic Differentiator
// author:  Vassil Vassilev <vvasilev-at-cern.ch>
//------------------------------------------------------------------------------

#include "GeneratedSourceFile.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/Compatibility.h"

#include "clang/AST/Decl.h"
#include "clang/AST/DeclBase.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/OperationKinds.h"
#include "clang/AST/PrettyPrinter.h"
#include "clang/AST/TemplateBase.h"
#include "clang/AST/Type.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/Specifiers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/ParsedTemplate.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <string>
#include <system_error>
#include <utility>

using namespace clang;

namespace clad {
/// Adds to \p TLI the template arguments of the clad::member_key that
/// identifies \p F: its class, its type and its position among the fields.
static void AddMemberKeyArgs(Sema& S, const FieldDecl* F,
                             TemplateArgumentListInfo& TLI) {
  ASTContext& C = S.getASTContext();
  SourceLocation Loc = utils::GetValidSLoc(S);
  // A member's type may use a typedef of its class, which can be private too,
  // so write the type it stands for.
  QualType FieldTy = F->getType();
  while (const auto* TT = FieldTy->getAs<TypedefType>()) {
    if (!TT->getDecl()->getDeclContext()->isRecord())
      break;
    FieldTy = TT->desugar();
  }
  for (QualType T : {clad_compat::getRecordType(C, F->getParent()), FieldTy}) {
    T = utils::makeTypeReadable(S, T);
    TLI.addArgument(TemplateArgumentLoc(TemplateArgument(T),
                                        C.getTrivialTypeSourceInfo(T, Loc)));
  }
  // Already converted, as a specialization's arguments are: Sema takes it as
  // it is.
  llvm::APSInt Id(llvm::APInt(C.getIntWidth(C.IntTy), F->getFieldIndex()),
                  /*isUnsigned=*/false);
  TLI.addArgument(S.getTrivialTemplateArgumentLoc(
      TemplateArgument(C, Id, C.IntTy), C.IntTy, Loc));
}

/// Makes \p F readable through clad::member. Unless the translation unit has it
/// already, adds this explicit instantiation at global scope to \p New:
///
///   template struct clad::member_access<clad::member_key<C, T, Id>, &C::m>;
///
/// An explicit instantiation may name members the derivative cannot access
/// ([temp.spec.general]). \returns whether \p F is readable.
static bool openMember(Sema& S, const FieldDecl* F,
                       llvm::SmallVectorImpl<Decl*>& New) {
  // Both are in Sins.h, which Differentiator.h includes; every derivative needs
  // Differentiator.h.
  TemplateDecl* Key = utils::LookupTemplateDeclInCladNamespace(S, "member_key");
  auto* Access = cast<ClassTemplateDecl>(
      utils::LookupTemplateDeclInCladNamespace(S, "member_access"));
  SourceLocation Loc = utils::GetValidSLoc(S);
  TemplateArgumentListInfo KeyArgs(Loc, Loc);
  AddMemberKeyArgs(S, F, KeyArgs);
  QualType KeyTy = utils::InstantiateTemplate(S, Key, KeyArgs);
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
  auto* Field = const_cast<FieldDecl*>(F);
  CXXScopeSpec ClassSS;
  utils::BuildNNS(S, Field->getParent(), ClassSS);
  Expr* Member = S.BuildUnaryOp(/*S=*/nullptr, Loc, UO_AddrOf,
                                S.BuildDeclRefExpr(Field, Field->getType(),
                                                   VK_LValue, Loc, &ClassSS))
                     .get();
  if (KeyTy.isNull() || !Member)
    return false;
  for (const ClassTemplateSpecializationDecl* Spec : Access->specializations())
    if (Spec->getSpecializationKind() == TSK_ExplicitInstantiationDefinition &&
        Spec->getTemplateArgs()[0].getAsType().getCanonicalType() ==
            KeyTy.getCanonicalType())
      return true;
  Sema::ContextRAII AtGlobalScope(S,
                                  S.getASTContext().getTranslationUnitDecl());
  CXXScopeSpec CladSS;
  utils::BuildNNS(S, utils::GetCladNamespace(S), CladSS);
  ParsedTemplateArgument Args[] = {
      ParsedTemplateArgument(ParsedTemplateArgument::Type,
                             ParsedType::make(KeyTy).getAsOpaquePtr(), Loc),
      ParsedTemplateArgument(ParsedTemplateArgument::NonType, Member, Loc)};
  DeclResult D = S.ActOnExplicitInstantiation(
      S.TUScope, /*ExternLoc=*/SourceLocation(), Loc, DeclSpec::TST_struct, Loc,
      CladSS, Sema::TemplateTy::make(TemplateName(Access)), Loc, Loc,
      ASTTemplateArgsPtr(Args), Loc, ParsedAttributesView::none());
  if (D.isInvalid())
    return false;
  New.push_back(D.get());
  return true;
}

/// Rewrites each read of a member that \p FD may not access into
/// clad::member<C, T, Id>(obj) and records the member in \p Opened. Members
/// that cannot be read this way are recorded in \p Unopened. \returns the
/// explicit instantiations of clad::member_access this needed.
static llvm::SmallVector<Decl*, 4>
readInaccessibleMembers(Sema& S, FunctionDecl* FD,
                        llvm::SmallVectorImpl<const FieldDecl*>& Opened,
                        llvm::SmallVectorImpl<const ValueDecl*>& Unopened) {
  llvm::SmallVector<Decl*, 4> New;
  // A derivative that is a member of the class has the same access as the
  // member it differentiates, so nothing needs rewriting.
  Stmt* Body = FD->getBody();
  if (!Body || isa<CXXMethodDecl>(FD))
    return New;
  Sema::ContextRAII InDerivative(S, FD);
  SourceLocation Loc = utils::GetValidSLoc(S);
  NamespaceDecl* CladNS = utils::GetCladNamespace(S);
  std::function<Stmt*(Stmt*)> Read = [&](Stmt* Node) -> Stmt* {
    for (Stmt*& Child : Node->children())
      if (Child)
        Child = Read(Child);
    auto* ME = dyn_cast<MemberExpr>(Node);
    if (!ME)
      return Node;
    ValueDecl* M = ME->getMemberDecl();
    // clad::member cannot read bit-fields or member functions; leave those as
    // they are.
    auto Unopen = [&]() {
      if (M->getAccess() != AS_public && M->getAccess() != AS_none &&
          !llvm::is_contained(Unopened, M))
        Unopened.push_back(M);
      return Node;
    };
    auto* F = dyn_cast<FieldDecl>(M);
    // A bit-field has no pointer to member to read it through.
    if (!F || F->isBitField() || F->getAccess() == AS_public)
      return Unopen();
    QualType ObjTy = ME->getBase()->getType();
    if (ME->isArrow())
      ObjTy = ObjTy->getPointeeType();
    CXXRecordDecl* NamingClass = ObjTy->getAsCXXRecordDecl();
    if (!NamingClass || S.IsSimplyAccessible(F, NamingClass, ObjTy) ||
        !openMember(S, F, New))
      return Unopen();
    // clad::member<C, T, Id>(obj)
    TemplateArgumentListInfo TLI(Loc, Loc);
    AddMemberKeyArgs(S, F, TLI);
    LookupResult R = utils::LookupQualifiedName("member", S, CladNS);
    CXXScopeSpec CladSS;
    utils::BuildNNS(S, CladNS, CladSS);
    Expr* Fn =
        utils::BuildDeclarationNameExpr(S, CladSS, R, /*ADL=*/false, &TLI);
    Expr* Obj = ME->getBase();
    if (ME->isArrow())
      Obj = S.BuildUnaryOp(/*S=*/nullptr, Loc, UO_Deref, Obj).get();
    Expr* Call = Fn && Obj
                     ? S.ActOnCallExpr(/*S=*/nullptr, Fn, Loc, Obj, Loc).get()
                     : nullptr;
    if (Call && !llvm::is_contained(Opened, F))
      Opened.push_back(F);
    return Call ? Call : Unopen();
  };
  FD->setBody(Read(Body));
  return New;
}

void GeneratedSourceFile::add(Sema& S, Decl* D, bool DeclarationOnly,
                              const FunctionDecl* Primal) {
  auto* FD = dyn_cast<FunctionDecl>(D);
  Entry E{D,
          DeclarationOnly,
          FD && Primal ? utils::GetSelfParam(FD, Primal) : nullptr,
          {},
          {}};
  // C++ does not let FD access some of the members it reads. Add the
  // instantiations that give access to them ahead of FD in the file.
  if (FD && !DeclarationOnly)
    for (Decl* Access : readInaccessibleMembers(S, FD, E.Opened, E.Unopened))
      m_Entries.push_back({Access, /*DeclarationOnly=*/false, nullptr, {}, {}});
  m_Entries.push_back(std::move(E));
}

/// What to print for \p D: the outermost namespace clad declared it in, a
/// clone of its function's own that holds only \p D, or \p D itself.
static const Decl* outermostNamespace(const Decl* D) {
  const Decl* Top = D;
  for (const DeclContext* DC = D->getLexicalDeclContext();
       DC && isa<NamespaceDecl>(DC);
       DC = cast<NamespaceDecl>(DC)->getLexicalDeclContext())
    Top = cast<NamespaceDecl>(DC);
  return Top;
}

/// The declaration of \p FD as a member of the class its `self` points to, for
/// the user to add to that class.
static std::string memberDeclaration(const FunctionDecl* FD,
                                     const PrintingPolicy& Policy) {
  std::string Decl;
  llvm::raw_string_ostream OS(Decl);
  FD->getReturnType().print(OS, Policy);
  OS << " " << FD->getName() << "(";
  for (unsigned I = 1, N = FD->getNumParams(); I < N; ++I) {
    if (I > 1)
      OS << ", ";
    FD->getParamDecl(I)->print(OS, Policy);
  }
  OS << ")";
  if (FD->getParamDecl(0)->getType()->getPointeeType().isConstQualified())
    OS << " const";
  OS << ";";
  return Decl;
}

/// Warns about the parts of \p D that do not compile as source: a member clad
/// added to a class (C++ cannot add members from outside a class), and the
/// members a free derivative reads through clad::member because it may not
/// access them. If the user declares the derivative in the class, it defines
/// that member and reads them directly.
static void diagnoseUnwritable(DiagnosticsEngine& Diags, const Decl* D,
                               const ParmVarDecl* Self,
                               llvm::ArrayRef<const FieldDecl*> Opened,
                               llvm::ArrayRef<const ValueDecl*> Unopened,
                               const PrintingPolicy& Policy) {
  const auto* FD = dyn_cast<FunctionDecl>(D);
  if (!FD)
    return;
  if (const auto* MD = dyn_cast<CXXMethodDecl>(FD)) {
    if (MD->getParent()->isLambda()) {
      unsigned ID = Diags.getCustomDiagID(
          DiagnosticsEngine::Warning,
          "the derivative '%0' is a member clad adds to a lambda's closure "
          "type; read back as source it does not compile");
      Diags.Report(MD->getLocation(), ID) << MD->getName();
    } else if (isa<CXXRecordDecl>(MD->getLexicalDeclContext())) {
      unsigned ID = Diags.getCustomDiagID(
          DiagnosticsEngine::Warning,
          "the derivative '%0' is a member clad adds to '%1'; read back as "
          "source it does not compile: declare it in the class");
      Diags.Report(MD->getLocation(), ID)
          << MD->getName() << MD->getParent()->getName();
    }
    return;
  }
  if (Opened.empty() && Unopened.empty())
    return;
  unsigned Warning = Diags.getCustomDiagID(
      DiagnosticsEngine::Warning,
      "the derivative '%0' reads '%1', which is %select{private|protected}2 in "
      "'%3', through an explicit instantiation of clad::member_access");
  for (const FieldDecl* F : Opened)
    Diags.Report(FD->getLocation(), Warning)
        << FD->getName() << F->getName() << (F->getAccess() == AS_protected)
        << F->getParent()->getQualifiedNameAsString();
  unsigned Unwritable = Diags.getCustomDiagID(
      DiagnosticsEngine::Warning,
      "the derivative '%0' reads '%1', which is not public in '%2'; read back "
      "as source it does not compile");
  for (const ValueDecl* M : Unopened)
    Diags.Report(FD->getLocation(), Unwritable)
        << FD->getName() << M->getNameAsString()
        << cast<NamedDecl>(M->getDeclContext())->getQualifiedNameAsString();
  // A free derivative of a member takes the object as its first parameter,
  // `self`.
  const CXXRecordDecl* RD =
      Self ? Self->getType()->getPointeeCXXRecordDecl() : nullptr;
  if (!RD)
    return;
  unsigned Note = Diags.getCustomDiagID(
      DiagnosticsEngine::Note, "to read %select{it|them}0 directly, declare "
                               "the derivative in '%1'%select"
                               "{|, written for this specialization}2: %3");
  Diags.Report(FD->getLocation(), Note)
      << (Opened.size() + Unopened.size() > 1) << RD->getName()
      << isa<ClassTemplateSpecializationDecl>(RD)
      << memberDeclaration(FD, Policy);
}

void GeneratedSourceFile::write(CompilerInstance& CI, llvm::StringRef Path) {
  // What a translation unit that did not compile made is not worth writing.
  if (m_Entries.empty() || CI.getDiagnostics().hasErrorOccurred()) {
    m_Entries.clear();
    return;
  }
  std::error_code EC;
  llvm::raw_fd_ostream Out(Path, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    unsigned ID = CI.getDiagnostics().getCustomDiagID(
        DiagnosticsEngine::Warning,
        "could not write the derivatives to '%0': %1");
    CI.getDiagnostics().Report(ID) << Path << EC.message();
    m_Entries.clear();
    return;
  }
  LangOptions LangOpts;
  LangOpts.CPlusPlus = true;
  PrintingPolicy Policy(LangOpts);
  Policy.Bool = true;
  // `ns::(anonymous namespace)::T` is no name; `ns::T` finds T.
  Policy.SuppressUnwrittenScope = true;
  for (const Entry& E : m_Entries) {
    if (!E.DeclarationOnly)
      diagnoseUnwritable(CI.getDiagnostics(), E.D, E.Self, E.Opened, E.Unopened,
                         Policy);
    const Decl* Top = outermostNamespace(E.D);
    clad_compat::printDecl(Top, Out, Policy);
    if (Top == E.D && E.DeclarationOnly)
      Out << ";";
    Out << "\n";
  }
  m_Entries.clear();
}
} // namespace clad
