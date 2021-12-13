#ifndef CLAD_DERIVED_TYPES_HANDLER_H
#define CLAD_DERIVED_TYPES_HANDLER_H

#include "clang/AST/Type.h"
#include "llvm/ADT/StringRef.h"
#include "clad/Differentiator/DerivedTypeEssentials.h"

#include <map>
#include <string>

namespace clang {
  class ASTConsumer;
  class CXXRecordDecl;
  class Sema;
} // namespace clang

namespace clad {
  class DerivedTypesHandler {
    clang::ASTConsumer& m_Consumer;
    clang::ASTContext& m_Context;
    clang::Sema& m_Sema;
    std::map<std::string, DerivedTypeEssentials> m_DerivedTypesEssentials;
    void SetDTE(llvm::StringRef name, DerivedTypeEssentials DTE);

  public:
    DerivedTypesHandler(clang::ASTConsumer& consumer, clang::Sema& semaRef);
    void InitialiseDerivedType(clang::QualType yQType, clang::QualType xQType,
                               clang::CXXRecordDecl* RD);
    DerivedTypeEssentials GetDTE(llvm::StringRef name);
  };
} // namespace clad

#endif