#include "clad/Differentiator/DerivedTypeEssentials.h"

#include "clang/AST/DeclBase.h"
#include "clang/AST/Decl.h"

using namespace clang;

namespace clad {

  void
  DerivedTypeEssentials::ProcessTopLevelDeclarations(ASTConsumer& consumer) {
    auto processTopLevelDecl = [&consumer](Decl* D) {
      if (!D)
        return;
      bool isTU = D->getDeclContext()->isTranslationUnit();
      if (isTU)
        consumer.HandleTopLevelDecl(DeclGroupRef(D));
    };
    processTopLevelDecl(m_DerivedAddFn);
    processTopLevelDecl(m_DerivedSubFn);
    processTopLevelDecl(m_DerivedMultiplyFn);
    processTopLevelDecl(m_DerivedDivideFn);
  }
} // namespace clad