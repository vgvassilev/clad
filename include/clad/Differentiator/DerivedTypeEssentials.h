#ifndef CLAD_DERIVED_TYPE_ESSENTIALS
#define CLAD_DERIVED_TYPE_ESSENTIALS

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/Type.h"

namespace clang {
  class FunctionDecl;
  class CXXMethodDecl;
  class CXXRecordDecl;
} // namespace clang
namespace clad {
  class DerivedTypeEssentials {
    clang::CXXRecordDecl* m_DerivedRD = nullptr;
    clang::FunctionDecl* m_DerivedAddFn = nullptr;
    clang::CXXMethodDecl* m_InitialiseSeedsFn = nullptr;
    clang::FunctionDecl* m_DerivedSubFn = nullptr;
    clang::FunctionDecl* m_DerivedMultiplyFn = nullptr;
    clang::FunctionDecl* m_DerivedDivideFn = nullptr;
    clang::QualType m_YQType, m_XQType;

  public:
    DerivedTypeEssentials(clang::QualType yQType = clang::QualType(),
                          clang::QualType xQType = clang::QualType(),
                          clang::CXXRecordDecl* derivedRD = nullptr,
                          clang::FunctionDecl* derivedAddFn = nullptr,
                          clang::FunctionDecl* derivedSubFn = nullptr,
                          clang::FunctionDecl* derivedMultiplyFn = nullptr,
                          clang::FunctionDecl* derivedDivideFn = nullptr,
                          clang::CXXMethodDecl* initialiseSeedsFn = nullptr)
        : m_YQType(yQType), m_XQType(xQType), m_DerivedRD(derivedRD),
          m_DerivedAddFn(derivedAddFn), m_DerivedSubFn(derivedSubFn),
          m_DerivedMultiplyFn(derivedMultiplyFn),
          m_DerivedDivideFn(derivedDivideFn),
          m_InitialiseSeedsFn(initialiseSeedsFn) {}
    clang::CXXMethodDecl* GetInitialiseSeedsFn() { return m_InitialiseSeedsFn; }
    clang::FunctionDecl* GetDerivedAddFn() { return m_DerivedAddFn; }
    clang::FunctionDecl* GetDerivedSubFn() { return m_DerivedSubFn; }
    clang::FunctionDecl* GetDerivedMultiplyFn() { return m_DerivedMultiplyFn; }
    clang::FunctionDecl* GetDerivedDivideFn() { return m_DerivedDivideFn; }
    clang::CXXRecordDecl* GetDerivedRD() { return m_DerivedRD; }
    bool isValid() const { return m_DerivedRD != nullptr; }
    clang::QualType GetYQType() const { return m_YQType; }
  };

} // namespace clad

#endif