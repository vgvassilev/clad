#ifndef CLAD_DERIVED_TYPES_HANDLER_H
#define CLAD_DERIVED_TYPES_HANDLER_H

#include "clang/AST/Type.h"
#include "clang/Basic/Version.h"
#include "llvm/ADT/StringRef.h"
#include "clad/Differentiator/DerivedTypeEssentials.h"

#include <map>
#include <string>
#include <utility>

namespace clang {
  class ASTConsumer;
  class CXXRecordDecl;
  class Sema;
} // namespace clang
namespace clang {
#if CLANG_VERSION_MAJOR <= 9
  static inline bool operator<(const clang::QualType lhs,
                               const clang::QualType rhs) {
    return lhs.getTypePtr() < rhs.getTypePtr();
  }
#endif
} // namespace clang
namespace clad {
  class DerivedTypesHandler {
    clang::ASTConsumer& m_Consumer;
    clang::ASTContext& m_Context;
    clang::Sema& m_Sema;
    std::map<std::pair<clang::QualType, clang::QualType>, DerivedTypeEssentials>
        m_DerivedTypesEssentials;
    void SetDTE(clang::QualType yType, clang::QualType xType,
                DerivedTypeEssentials DTE);

  public:
    DerivedTypesHandler(clang::ASTConsumer& consumer, clang::Sema& semaRef);
    void InitialiseDerivedType(clang::QualType yQType, clang::QualType xQType);
    DerivedTypeEssentials GetDTE(clang::QualType);
    clang::QualType GetDerivedType(clang::QualType yQType,
                                   clang::QualType xQType);
    clang::QualType GetYType(clang::QualType derivedQType);
  };
} // namespace clad

#endif