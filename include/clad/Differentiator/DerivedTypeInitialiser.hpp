#ifndef CLAD_DERIVED_TYPE_INITIALISER_H
#define CLAD_DERIVED_TYPE_INITIALISER_H

#include "clang/AST/Type.h"
#include "llvm/ADT/SmallVector.h"
#include "clad/Differentiator/ASTHelper.h"

#include <map>

namespace clang {
  class ASTContext;
  class CXXRecordDecl;
  class Expr;
  class FunctionDecl;
  class NamespaceDecl;
  class ParamVarDecl;
  class Sema;
} // namespace clang

namespace clad {
  class DerivedTypeEssentials {
    clang::FunctionDecl* m_DerivedAddFn=nullptr;
    public:
    DerivedTypeEssentials() = default;
    DerivedTypeEssentials(clang::FunctionDecl* derivedAddFn) : m_DerivedAddFn(derivedAddFn) {}
    clang::FunctionDecl* GetDerivedAddFnDecl() {
      return m_DerivedAddFn;
    }
  };

  class DerivedTypeInitialiser {
    clang::Sema& m_Sema;
    clang::ASTContext& m_Context;
    ASTHelper m_ASTHelper;
    clang::QualType m_yQType;
    clang::QualType m_xQType;
    clang::CXXRecordDecl* m_DerivedRecord = nullptr;
    clang::QualType m_DerivedType;
    std::map<std::string, clang::Expr*> m_Variables;
    clang::FunctionDecl* m_DerivedAddFnDecl;
  public:
    DerivedTypeInitialiser(clang::Sema& semaRef, clang::QualType yQType,
                           clang::QualType xQType,
                           clang::CXXRecordDecl* derivedRecord);
    DerivedTypeEssentials CreateDerivedTypeEssentials();

  private:
    // bool FillDerivedRecord();
    // bool CreateInitialiseSeedsFn();
    clang::QualType GetDerivedParamType() const;
    clang::QualType GetNonDerivedParamType() const;
    clang::NamespaceDecl* BuildCladNamespace();
    clang::FunctionDecl* CreateDerivedAddFn();
    // bool CreateDerivedSubFn();
    clang::QualType ComputeDerivedAddSubFnType() const;
    llvm::SmallVector<clang::ParmVarDecl*, 2> CreateDerivedAddSubFnParams();
  };
} // namespace clad

#endif