#include "clad/Differentiator/DerivedTypesHandler.h"

#include "clad/Differentiator/CladUtils.h"
#include "clad/Differentiator/DerivedTypeInitialiser.h"

#include "clang/Sema/Sema.h"

using namespace clang;

namespace clad {
  DerivedTypesHandler::DerivedTypesHandler(ASTConsumer& consumer, Sema& semaRef)
      : m_Consumer(consumer), m_Sema(semaRef),
        m_Context(semaRef.getASTContext()) {}

  void DerivedTypesHandler::SetDTE(llvm::StringRef name,
                                   DerivedTypeEssentials DTE) {
    m_DerivedTypesEssentials[name] = DTE;
  }

  void DerivedTypesHandler::InitialiseDerivedType(QualType yQType,
                                                  QualType xQType,
                                                  clang::CXXRecordDecl* RD) {
    auto DTI = DerivedTypeInitialiser(m_Consumer, m_Sema, *this, yQType, xQType,
                                      RD);
    auto DTE = DTI.CreateDerivedTypeEssentials();
    SetDTE(RD->getName(), DTE);
  }

  DerivedTypeEssentials DerivedTypesHandler::GetDTE(llvm::StringRef name) {
    auto it = m_DerivedTypesEssentials.find(name);
    if (it == m_DerivedTypesEssentials.end()) {
      return DerivedTypeEssentials();
    }
    return it->second;
  }

  QualType DerivedTypesHandler::GetDerivedType(clang::QualType yQType,
                                               clang::QualType xQType) {
    auto yTypeName = yQType.getAsString();
    if (yQType->isClassType())
      yTypeName = utils::GetRecordName(yQType);
    auto xTypeName = xQType.getAsString();
    if (xQType->isClassType())
      xTypeName = utils::GetRecordName(xQType);
    auto name = "__clad_" + yTypeName + "_wrt_" + xTypeName;
    auto it = m_DerivedTypesEssentials.find(name);
    if (it != m_DerivedTypesEssentials.end()) {
      return it->second.GetDerivedRD()
          ->getTypeForDecl()
          ->getCanonicalTypeInternal();
    }
    return QualType();
  }
} // namespace clad