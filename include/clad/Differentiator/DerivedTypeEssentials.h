#ifndef CLAD_DERIVED_TYPE_ESSENTIALS
#define CLAD_DERIVED_TYPE_ESSENTIALS

namespace clang {
  class FunctionDecl;
}
namespace clad {
  class DerivedTypeEssentials {
    clang::FunctionDecl* m_DerivedAddFn = nullptr;

  public:
    DerivedTypeEssentials(clang::FunctionDecl* derivedAddFn = nullptr)
        : m_DerivedAddFn(derivedAddFn) {}
    clang::FunctionDecl* GetDerivedAddFnDecl() { return m_DerivedAddFn; }
  };
} // namespace clad

#endif