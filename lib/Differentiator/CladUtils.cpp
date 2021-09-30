#include "clad/Differentiator/CladUtils.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Lookup.h"
#include "clad/Differentiator/Compatibility.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;
namespace clad {
  namespace utils {
    static SourceLocation noLoc{};
    
    std::string ComputeEffectiveFnName(const FunctionDecl* FD) {
      // TODO: Add cases for more operators
      switch (FD->getOverloadedOperator()) {
        case OverloadedOperatorKind::OO_Call: return "operator_call";
        default: return FD->getNameAsString();
      }
    }

    CompoundStmt* PrependAndCreateCompoundStmt(ASTContext& C, Stmt* initial,
                                               Stmt* S) {
      llvm::SmallVector<Stmt*, 16> block;
      block.push_back(S);
      if (CompoundStmt* CS = dyn_cast<CompoundStmt>(initial))
        block.append(CS->body_begin(), CS->body_end());
      else 
        block.push_back(initial);
      auto stmtsRef = llvm::makeArrayRef(block.begin(), block.end());
      return clad_compat::CompoundStmt_Create(C, stmtsRef, noLoc, noLoc);
    }

    CompoundStmt* AppendAndCreateCompoundStmt(ASTContext& C, Stmt* initial,
                                              Stmt* S) {
      llvm::SmallVector<Stmt*, 16> block;
      assert(isa<CompoundStmt>(initial) &&
             "initial should be of type `clang::CompoundStmt`");
      if (CompoundStmt* CS = dyn_cast<CompoundStmt>(initial))
        block.append(CS->body_begin(), CS->body_end());
      block.push_back(S);
      auto stmtsRef = llvm::makeArrayRef(block.begin(), block.end());
      return clad_compat::CompoundStmt_Create(C, stmtsRef, noLoc, noLoc);
    }

    void BuildNNS(clang::Sema& semaRef, DeclContext* DC, CXXScopeSpec& CSS,
                  bool addGlobalNS) {
      assert(DC && "Must provide a non null DeclContext");

      // parent name specifier should be added first
      if (DC->getParent())
        BuildNNS(semaRef, DC->getParent(), CSS);

      ASTContext& C = semaRef.getASTContext();

      if (auto ND = dyn_cast<NamespaceDecl>(DC)) {
        CSS.Extend(C, ND,
                   /*NamespaceLoc=*/noLoc,
                   /*ColonColonLoc=*/noLoc);
      } else if (auto RD = dyn_cast<CXXRecordDecl>(DC)) {
        auto RDQType = RD->getTypeForDecl()->getCanonicalTypeInternal();
        auto RDTypeSourceInfo = C.getTrivialTypeSourceInfo(RDQType);
        CSS.Extend(C,
                   /*TemplateKWLoc=*/noLoc, RDTypeSourceInfo->getTypeLoc(),
                   /*ColonColonLoc=*/noLoc);
      } else if (addGlobalNS && isa<TranslationUnitDecl>(DC)) {
        CSS.MakeGlobal(C, /*ColonColonLoc=*/noLoc);
      }
    }

    DeclContext* FindDeclContext(clang::Sema& semaRef, clang::DeclContext* DC1,
                                 clang::DeclContext* DC2) {
      // llvm::errs()<<"DC1 name: "<<DC1->getDeclKindName()<<"\n";
      // llvm::errs()<<"DC2 name: "<<DC2->getDeclKindName()<<"\n";
      // cast<Decl>(DC1)->dumpColor();
      llvm::SmallVector<clang::DeclContext*, 4> contexts;
      assert((isa<NamespaceDecl>(DC1) || isa<TranslationUnitDecl>(DC1)) &&
             "DC1 can only be extended if it is a "
             "namespace or translation unit decl.");
      while (DC2) {
        // llvm::errs()<<"DC2 name: "<<DC2->getDeclKindName()<<"\n";
        if (isa<TranslationUnitDecl>(DC2))
          break;
        if (isa<LinkageSpecDecl>(DC2)) {
          DC2 = DC2->getParent();  
          continue;
        }
        assert(isa<NamespaceDecl>(DC2) &&
               "DC2 should only contain namespace (and "
               "translation unit) declaration.");
        contexts.push_back(DC2);
        DC2 = DC2->getParent();
      }
      DeclContext* DC = DC1;
      for (int i = contexts.size() - 1; i >= 0; --i) {
        NamespaceDecl* ND = cast<NamespaceDecl>(contexts[i]);
        DC = LookupNSD(semaRef, ND->getIdentifier()->getName(),
                       /*shouldExist=*/false, DC1);
        if (!DC)
          return nullptr;
        DC1 = DC;
      }
      return DC->getPrimaryContext();
    }

    NamespaceDecl* LookupNSD(Sema& S, llvm::StringRef namespc, bool shouldExist,
                             DeclContext* DC) {
      ASTContext& C = S.getASTContext();
      if (!DC)
        DC = C.getTranslationUnitDecl();
      // Find the builtin derivatives/numerical diff namespace
      DeclarationName Name = &C.Idents.get(namespc);
      LookupResult R(S, Name, SourceLocation(), Sema::LookupNamespaceName,
                     clad_compat::Sema_ForVisibleRedeclaration);
      S.LookupQualifiedName(R, DC,
                            /*allowBuiltinCreation*/ false);
      if (!shouldExist && R.empty())
        return nullptr;
      assert(!R.empty() && "Cannot find the specified namespace!");
      NamespaceDecl* ND = cast<NamespaceDecl>(R.getFoundDecl());
      return cast<NamespaceDecl>(ND->getPrimaryContext());
    }

    clang::DeclContext* GetOutermostDC(Sema& semaRef, clang::DeclContext* DC) {
      ASTContext& C = semaRef.getASTContext();
      assert(DC && "Invalid DC");
      while (DC) {
        if (DC->getParent() == C.getTranslationUnitDecl())
          break;
        DC = DC->getParent();
      }
      return DC;
    }
    
    StringLiteral* CreateStringLiteral(ASTContext& C, llvm::StringRef str) {
      // Copied and adapted from clang::Sema::ActOnStringLiteral.
      QualType CharTyConst = C.CharTy.withConst();
      QualType
          StrTy = clad_compat::getConstantArrayType(C, CharTyConst,
                                                    llvm::APInt(/*numBits=*/32,
                                                                str.size() + 1),
                                                    /*SizeExpr=*/nullptr,
                                                    /*ASM=*/ArrayType::Normal,
                                                    /*IndexTypeQuals*/ 0);
      StringLiteral* SL = StringLiteral::Create(C, str,
                                                /*Kind=*/StringLiteral::Ascii,
                                                /*Pascal=*/false, StrTy, noLoc);
      return SL;
    }

    bool isArrayOrPointerType(const clang::QualType QT) {
      return QT->isArrayType() || QT->isPointerType();
    }
  } // namespace utils
} // namespace clad