#ifndef CLAD_DERIVED_TYPE_ESSENTIALS
#define CLAD_DERIVED_TYPE_ESSENTIALS

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/DeclGroup.h"

namespace clang {
  class FunctionDecl;
  class CXXMethodDecl;
}
namespace clad {
  class DerivedTypeEssentials {
    clang::FunctionDecl* m_DerivedAddFn = nullptr;
    clang::CXXMethodDecl* m_InitialiseSeedsFn = nullptr;
    clang::FunctionDecl* m_DerivedSubFn = nullptr;
    clang::FunctionDecl* m_DerivedMultiplyFn = nullptr;
    clang::FunctionDecl* m_DerivedDivideFn = nullptr;

  public:
    DerivedTypeEssentials(clang::FunctionDecl* derivedAddFn = nullptr,
                          clang::FunctionDecl* derivedSubFn = nullptr,
                          clang::FunctionDecl* derivedMultiplyFn = nullptr,
                          clang::FunctionDecl* derivedDivideFn = nullptr,
                          clang::CXXMethodDecl* initialiseSeedsFn = nullptr)
        : m_DerivedAddFn(derivedAddFn), m_DerivedSubFn(derivedSubFn),
          m_DerivedMultiplyFn(derivedMultiplyFn),
          m_DerivedDivideFn(derivedDivideFn),
          m_InitialiseSeedsFn(initialiseSeedsFn) {}
    clang::CXXMethodDecl* GetInitialiseSeedsFn() { return m_InitialiseSeedsFn; }
    clang::FunctionDecl* GetDerivedAddFn() { return m_DerivedAddFn; }
    clang::FunctionDecl* GetDerivedSubFn() { return m_DerivedSubFn; }
    clang::FunctionDecl* GetDerivedMultiplyFn() { return m_DerivedMultiplyFn; }
    clang::FunctionDecl* GetDerivedDivideFn() { return m_DerivedDivideFn; }
  
    void ProcessTopLevelDeclarations(clang::ASTConsumer& consumer);
  };

} // namespace clad

#endif