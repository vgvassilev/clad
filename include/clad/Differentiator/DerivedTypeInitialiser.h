#ifndef CLAD_DERIVED_TYPE_INITIALISER_H
#define CLAD_DERIVED_TYPE_INITIALISER_H

#include "clang/AST/Type.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Sema/Scope.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "clad/Differentiator/ASTHelper.h"
#include "clad/Differentiator/DerivedTypeEssentials.h"

#include <map>

namespace clang {
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

  class DerivedTypeInitialiser {
    clang::Sema& m_Sema;
    clang::ASTContext& m_Context;
    ASTHelper m_ASTHelper;
    clang::QualType m_yQType;
    clang::QualType m_xQType;
    clang::CXXRecordDecl* m_DerivedRecord = nullptr;
    clang::QualType m_DerivedType;
    std::map<std::string, clang::Expr*> m_Variables;
    clang::FunctionDecl* m_DerivedAddFn = nullptr;
    clang::FunctionDecl* m_DerivedSubFn = nullptr;
    clang::FunctionDecl* m_DerivedMultiplyFn = nullptr;
    clang::FunctionDecl* m_DerivedDivideFn = nullptr;
    clang::CXXMethodDecl* m_InitialiseSeedsFn = nullptr;
    clang::Scope* m_CurScope;
  public:
    DerivedTypeInitialiser(clang::Sema& semaRef, clang::QualType yQType,
                           clang::QualType xQType,
                           clang::CXXRecordDecl* derivedRecord);
    DerivedTypeEssentials CreateDerivedTypeEssentials();

  private:
    void FillDerivedRecord();
    clang::CXXMethodDecl* CreateInitialiseSeedsFn();
    clang::QualType GetDerivedParamType() const;
    clang::QualType GetNonDerivedParamType() const;
    clang::NamespaceDecl* BuildCladNamespace();
    clang::FunctionDecl* CreateDerivedAddFn();
    clang::FunctionDecl* BuildDerivedSubFn();
    // bool CreateDerivedSubFn();
    clang::QualType ComputeDerivedAddSubFnType() const;
    clang::QualType ComputeInitialiseSeedsFnType() const;
    llvm::SmallVector<clang::ParmVarDecl*, 2> CreateDerivedAddSubFnParams();
    void ComputeAndStoreDRE(llvm::ArrayRef<clang::ValueDecl*> decls);
    void beginScope(unsigned);
    void endScope();
  };
} // namespace clad

#endif