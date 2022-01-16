#ifndef CLAD_DERIVED_TYPE_INITIALISER_H
#define CLAD_DERIVED_TYPE_INITIALISER_H

#include "clang/AST/DeclarationName.h"
#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "clad/Differentiator/ASTHelper.h"
#include "clad/Differentiator/DerivedTypeEssentials.h"

#include <map>
#include <string>
#include <vector>

namespace clang {
  class ASTConsumer;
  class ASTContext;
  class CXXRecordDecl;
  class DeclarationNameInfo;
  class Expr;
  class FunctionDecl;
  class NamespaceDecl;
  class ParamVarDecl;
  class Sema;
  class ValueDecl;
} // namespace clang

namespace clad {
  class DerivedTypesHandler;
  class DerivedTypeInitialiser {
    clang::Sema& m_Sema;
    clang::ASTContext& m_Context;
    DerivedTypesHandler& m_DTH;
    ASTHelper m_ASTHelper;
    clang::QualType m_YQType;
    clang::QualType m_XQType;
    clang::ClassTemplateSpecializationDecl* m_TangentRD = nullptr;
    clang::QualType m_TangentQType;

    std::map<std::string, clang::Expr*> m_Variables;
    clang::FunctionDecl* m_DerivedAddFn = nullptr;
    clang::FunctionDecl* m_DerivedSubFn = nullptr;
    clang::FunctionDecl* m_DerivedMultiplyFn = nullptr;
    clang::FunctionDecl* m_DerivedDivideFn = nullptr;
    clang::CXXMethodDecl* m_InitialiseSeedsFn = nullptr;
    clang::ClassTemplateSpecializationDecl* m_DerivedTypeInfoSpec = nullptr;
    clang::Scope* m_CurScope;
    
    using Stmts = llvm::SmallVector<clang::Stmt*, 16>;
    std::vector<Stmts> m_Blocks;

  public:
    DerivedTypeInitialiser(clang::ASTConsumer& consumer, clang::Sema& semaRef,
                           DerivedTypesHandler& DTH, clang::QualType yQType,
                           clang::QualType xQType);
    DerivedTypeEssentials CreateDerivedTypeEssentials();

  private:
    void BuildTangentDefinition();
    void FillDerivedRecord();
    void InitialiseIndependentFields(clang::Expr* base, llvm::SmallVector<std::string, 4> path);
    clang::CXXMethodDecl* BuildInitialiseSeedsFn();
    clang::QualType GetDerivedParamType() const;
    clang::QualType GetNonDerivedParamType() const;
    clang::NamespaceDecl* BuildCladNamespace();
    clang::FunctionDecl* BuildDerivedAddFn();
    clang::FunctionDecl* BuildDerivedSubFn();
    clang::FunctionDecl* BuildDerivedMultiplyFn();
    clang::FunctionDecl* BuildDerivedDivideFn();
    template <class ComputeDerivedFnTypeT, class BuildDerivedFnParamsT,
              class BuildFnBodyT>
    clang::FunctionDecl*
    GenerateDerivedArithmeticFn(clang::DeclarationName fnDName,
                                ComputeDerivedFnTypeT ComputeDerivedFnType,
                                BuildDerivedFnParamsT BuildDerivedFnParams,
                                BuildFnBodyT buildFnBody);
    clang::QualType ComputeDerivedAddSubFnType() const;
    clang::QualType ComputeDerivedMultiplyDivideFnType();
    clang::QualType ComputeInitialiseSeedsFnType() const;
    llvm::SmallVector<clang::ParmVarDecl*, 2> BuildDerivedAddSubFnParams();
    llvm::SmallVector<clang::ParmVarDecl*, 4>
    BuildDerivedMultiplyDivideFnParams();
    void ComputeAndStoreDRE(llvm::ArrayRef<clang::ValueDecl*> decls);
    void beginScope(unsigned);
    void endScope();
    bool AddToCurrentBlock(clang::Stmt* S);
    Stmts& BeginBlock();
    clang::CompoundStmt* EndBlock();
    void ProcessTopLevelDeclarations(clang::ASTConsumer& consumer);
    clang::Scope* GetCurrentScope();
    clang::ClassTemplateDecl* FindDerivedTypeInfoBaseTemplate();
    clang::ClassTemplateDecl* FindDerivativeOfBaseTemplate();
    void BuildTangentTypeInfoSpecialisation();
  };
} // namespace clad

#endif